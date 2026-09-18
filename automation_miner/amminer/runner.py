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
from .llm.blueprint import candidate_to_automation, render_yaml
from .llm.equivalence import matches, semantic_form
from .llm.generate import GenerationResult, generate
from .llm.provider import build_provider
from .llm.validate import validate_automation
from .miners.base import Action, Candidate, Condition, Evidence, Trigger
from .pipeline import RunReport, run_analysis
from .store import Store

_LOGGER = logging.getLogger(__name__)


def generation_digest(config: dict[str, Any]) -> str:
    """Identity of what an automation *does*, for consent purposes.

    Keyed on the semantic form, so re-rendering the same rule yields the same
    digest while any change to trigger/condition/action changes it.
    """
    import hashlib
    import json as _json

    payload = _json.dumps(semantic_form(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
        #: Set whenever no analysis is in flight, so shutdown can wait for one.
        self._idle = threading.Event()
        self._idle.set()

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
            self._idle.clear()
        started = time.time()
        try:
            # A fresh resolver each run picks up newly added entities.
            self.resolver = build_resolver(self.options.ha_config_dir, self.client)
            self._services = None
            report, _candidates = run_analysis(
                self.options, self.store, self.client, self.ha_config, resolver=self.resolver
            )
            self.last_report = report
            return report
        except Exception as err:  # noqa: BLE001 - the service must survive
            _LOGGER.exception("Unhandled error during analysis: %s", err)
            return self.last_report
        finally:
            self._running = False
            self._idle.set()
            _LOGGER.info("Analysis finished in %.1fs", time.time() - started)

    def wait_until_idle(self, timeout: float) -> bool:
        """Block until no analysis is in flight.  False if it is still going.

        Shutdown closes the database.  Doing that under a run that is still
        writing raises "Cannot operate on a closed database" mid-write, loses
        every candidate not yet persisted, and leaves the run row saying
        "running" forever.
        """
        return self._idle.wait(timeout)

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
        # Persist exactly what is about to be rendered, so Apply writes this
        # artifact rather than asking the model the same question again.
        if result.config:
            self.store.save_generation(
                suggestion_id, generation_digest(result.config), result.source, result.config
            )
        data = result.as_dict()
        data["digest"] = generation_digest(result.config) if result.config else None
        return data

    def apply(
        self, suggestion_id: str, confirm_conflicts: bool = False
    ) -> dict[str, Any] | None:
        """Generate, fully validate (including check_config) and write."""
        stored = self.store.get_suggestion(suggestion_id)
        if stored is None:
            return None
        payload = stored.get("payload") or {}
        if not payload.get("actions"):
            return {"ok": False, "errors": ["This finding has no automation to apply."]}
        candidate = candidate_from_payload(payload)

        # A conflict of error severity means this rule and an existing one pull
        # the same entity opposite ways.  The conflict check was being run,
        # counted in the report, and then not consulted by the one operation it
        # exists to inform.  The user still decides - they just have to say so.
        blocking = [
            conflict
            for conflict in (payload.get("conflicts") or [])
            if conflict.get("severity") == "error"
        ]
        if blocking and not confirm_conflicts:
            return {
                "ok": False,
                "needs_confirmation": True,
                "conflicts": blocking,
                "errors": [
                    "This rule conflicts with an automation you already have: "
                    + "; ".join(str(c.get("message", "")) for c in blocking[:3])
                ],
            }

        # Apply what was reviewed. Regenerating here would ask a
        # non-deterministic model the same question a second time and write the
        # second answer - an automation the user never saw. The stored artifact
        # is still re-validated below, including check_config, before it is
        # written; reuse means "no new content", not "no new checks".
        stored = self.store.get_generation(suggestion_id)
        if stored is not None and not matches(
            stored["payload"], candidate_to_automation(candidate, self._ensure_resolver())
        ):
            # The finding was re-mined into a different rule after the preview.
            # The stored artifact is no longer what this suggestion says, so it
            # is not what the user approved either.
            stored = None

        if stored is not None:
            generation = GenerationResult()
            generation.config = stored["payload"]
            generation.source = f"{stored.get('source') or 'stored'} (reviewed)"
            generation.yaml_text = render_yaml(stored["payload"])
            generation.notes.append(
                "Applying the automation exactly as it was previewed."
            )
            generation.report = validate_automation(
                generation.config,
                resolver=self._ensure_resolver(),
                client=self.client if self.client.configured else None,
                known_services=self._ensure_services(),
                run_check_config=True,
            )
        else:
            # Never previewed (an API caller, or a store that lost the row):
            # generate once and apply that same object - still a single
            # generation, so there is no divergence window.
            generation = generate(
                candidate,
                resolver=self._ensure_resolver(),
                provider=build_provider(self.options),
                client=self.client if self.client.configured else None,
                services=self._ensure_services(),
                run_check_config=True,
            )
            generation.notes.append(
                "No stored preview for this suggestion; generated and applied in one step."
            )

        result = apply_generation(generation, self.client)
        if result.ok and result.automation_id:
            # The stable marker is result.automation_id - the "id" this add-on
            # just wrote into the automation itself (see apply.py), which Home
            # Assistant keeps fixed across a rename or an edit.  Snapshotting
            # both the neutral candidate (replayable by amminer.health via
            # candidate_from_payload, exactly as it was backtested) and the
            # exact config written (so a later edit can be detected) is what
            # lets a health check recognise and judge this automation without
            # ever re-asking whether it was applied.
            self.store.record_applied_automation(
                automation_id=result.automation_id,
                suggestion_id=suggestion_id,
                title=candidate.title,
                candidate_payload=candidate.as_dict(self._ensure_resolver()),
                shipped_config=dict(generation.config or {}),
            )
        data = result.as_dict()
        data["generation"] = generation.as_dict()
        return data
