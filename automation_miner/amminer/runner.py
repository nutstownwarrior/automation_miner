"""The long-lived service object: scheduling, run state, previews, applying.

The web layer talks only to this, so it never imports the pipeline directly and
stays easy to test with a stub.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .apply import apply_generation
from .config import Options
from .discovery.ha_config import HAConfig
from .entities import EntityResolver, build_resolver
from .ha_api import HAClient
from .llm.generate import generate
from .llm.provider import build_provider
from .miners.base import Action, Candidate, Condition, Evidence, Trigger
from .pipeline import RunReport, run_analysis
from .store import Store

_LOGGER = logging.getLogger(__name__)


def candidate_from_payload(payload: dict[str, Any]) -> Candidate:
    """Rebuild a :class:`Candidate` from its stored dict form."""
    return Candidate(
        miner=str(payload.get("miner", "unknown")),
        title=str(payload.get("title", "Suggestion")),
        description=str(payload.get("description", "")),
        triggers=[Trigger(**t) for t in payload.get("triggers", [])],
        conditions=[Condition(**c) for c in payload.get("conditions", [])],
        actions=[
            Action(
                service=a["service"],
                entity_id=a.get("entity_id"),
                data=a.get("data", {}) or {},
            )
            for a in payload.get("actions", [])
            if a.get("service")
        ],
        evidence=Evidence(
            **{
                k: v
                for k, v in (payload.get("evidence") or {}).items()
                if k in Evidence.__dataclass_fields__
            }
        ),
        score=float(payload.get("score", 0.0)),
        entities=list(payload.get("entities", [])),
        extra=payload.get("extra", {}) or {},
        backtest=payload.get("backtest"),
        conflicts=payload.get("conflicts", []) or [],
    )


class Runner:
    """Owns analysis runs and everything the UI needs afterwards."""

    def __init__(self, options: Options, store: Store, client: HAClient | None = None) -> None:
        self.options = options
        self.store = store
        self.client = client if client is not None else HAClient()
        self.ha_config = HAConfig(options.ha_config_dir)
        self.last_report: RunReport | None = None
        self.resolver: EntityResolver | None = None
        self._services: list[str] | None = None
        self._lock = threading.Lock()
        self._running = False

    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._running

    def resolver_stats(self) -> dict[str, Any]:
        return self.resolver.stats() if self.resolver else {}

    def _ensure_resolver(self) -> EntityResolver:
        if self.resolver is None:
            self.resolver = build_resolver(self.options.ha_config_dir, self.client)
        return self.resolver

    def _ensure_services(self) -> list[str]:
        if self._services is None:
            self._services = sorted(self.client.service_index()) if self.client.configured else []
        return self._services

    # ------------------------------------------------------------------
    def run_now(self) -> RunReport | None:
        """Run one analysis.  Safe to call from a thread; never raises."""
        with self._lock:
            if self._running:
                _LOGGER.info("Analysis already running; ignoring request")
                return self.last_report
            self._running = True
        started = time.time()
        try:
            # A fresh resolver each run picks up newly added entities.
            self.resolver = build_resolver(self.options.ha_config_dir, self.client)
            self._services = None
            report, _candidates = run_analysis(
                self.options, self.store, self.client, self.ha_config
            )
            self.last_report = report
            return report
        except Exception as err:  # noqa: BLE001 - the service must survive
            _LOGGER.exception("Unhandled error during analysis: %s", err)
            return self.last_report
        finally:
            self._running = False
            _LOGGER.info("Analysis finished in %.1fs", time.time() - started)

    # ------------------------------------------------------------------
    def preview_yaml(self, suggestion_id: str) -> dict[str, Any] | None:
        """Generate + validate the YAML for one suggestion (no side effects)."""
        stored = self.store.get_suggestion(suggestion_id)
        if stored is None:
            return None
        payload = stored.get("payload") or {}
        if not payload.get("actions"):
            return {
                "ok": False,
                "source": "none",
                "yaml": "",
                "notes": ["This is a housekeeping finding, not an automation."],
                "validation": None,
            }
        candidate = candidate_from_payload(payload)
        result = generate(
            candidate,
            resolver=self._ensure_resolver(),
            provider=build_provider(self.options),
            client=self.client if self.client.configured else None,
            services=self._ensure_services(),
            run_check_config=False,  # a preview must not hammer Core
        )
        return result.as_dict()

    def apply(self, suggestion_id: str) -> dict[str, Any] | None:
        """Generate, fully validate (including check_config) and write."""
        stored = self.store.get_suggestion(suggestion_id)
        if stored is None:
            return None
        payload = stored.get("payload") or {}
        if not payload.get("actions"):
            return {"ok": False, "errors": ["This finding has no automation to apply."]}
        candidate = candidate_from_payload(payload)
        generation = generate(
            candidate,
            resolver=self._ensure_resolver(),
            provider=build_provider(self.options),
            client=self.client if self.client.configured else None,
            services=self._ensure_services(),
            run_check_config=True,
        )
        result = apply_generation(generation, self.client)
        data = result.as_dict()
        data["generation"] = generation.as_dict()
        return data
