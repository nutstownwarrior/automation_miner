"""Add-on options: defaults, loading from ``/data/options.json``, validation.

Every option has a sensible default so the add-on runs plug-and-play with an
empty ``options.json``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

#: Entities that are pure noise for behaviour mining.
DEFAULT_EXCLUDED_ENTITIES: tuple[str, ...] = (
    "sensor.time",
    "sensor.date",
    "sensor.date_time",
    "sensor.date_time_iso",
    "sensor.time_date",
    "sensor.time_utc",
    "sensor.uptime",
    "sensor.last_boot",
    "sensor.home_assistant_v2_db_size",
)

#: Domains that never represent a user *action* worth automating.
DEFAULT_EXCLUDED_DOMAINS: tuple[str, ...] = (
    "persistent_notification",
    "update",
    "sun",
    "tts",
    "stt",
    "conversation",
    "assist_satellite",
)

#: Domains whose state changes we treat as controllable actions.
ACTIONABLE_DOMAINS: tuple[str, ...] = (
    "light",
    "switch",
    "fan",
    "cover",
    "climate",
    "media_player",
    "lock",
    "vacuum",
    "humidifier",
    "water_heater",
    "input_boolean",
    "input_number",
    "input_select",
    "select",
    "number",
    "siren",
    "valve",
    "lawn_mower",
    "scene",
    "script",
)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


@dataclass
class Options:
    """Runtime options for a mining run."""

    log_level: str = "info"
    schedule: str = "0 3 * * *"
    run_on_start: bool = True

    # 0 == auto: min(available_history, 60 days)
    analysis_window_days: int = 0
    max_auto_window_days: int = 60

    # --- time-of-day miner ---
    min_consistency: float = 0.6
    min_occurrences: int = 5
    time_cluster_minutes: int = 30

    # --- association / sequence miners ---
    min_support: float = 0.02
    min_confidence: float = 0.6
    min_lift: float = 1.5
    association_window_seconds: int = 300
    sequence_min_occurrences: int = 4

    # --- override detection ---
    override_window_seconds: int = 120

    # --- backtesting gate ---
    backtest_min_precision: float = 0.7
    backtest_max_false_fires_per_week: float = 3.0
    backtest_match_tolerance_seconds: int = 900

    # --- staleness ---
    stale_automation_days: int = 30

    # --- exclusions ---
    excluded_domains: list[str] = field(default_factory=list)
    excluded_entities: list[str] = field(default_factory=list)
    excluded_users: list[str] = field(default_factory=list)

    # --- LLM ---
    llm_provider: str = "none"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_timeout_seconds: int = 180

    # --- optional AI assistance (all OFF by default) ---
    # Each of these needs llm_provider to be set to something other than
    # "none".  With the provider disabled they no-op and say so in the run
    # report; nothing below can make the add-on produce a suggestion that has
    # not passed the same deterministic gates as every other suggestion.
    #
    # Let the model read the entity inventory and label signal roles the
    # regex detector missed.  Additive only: it can add roles, never remove one
    # the deterministic detector found.
    llm_entity_classification: bool = False
    llm_classification_batch: int = 60
    # Let the model propose extra conditions for rules the backtest rejected.
    # Every proposal is re-backtested; only proposals that pass are surfaced.
    llm_hypotheses: bool = False
    llm_hypothesis_candidates: int = 10
    llm_hypotheses_per_candidate: int = 3
    # Let the model flag statistically real but semantically absurd rules.
    # Advisory only: it can demote and annotate, never promote or remove.
    llm_triage: bool = False
    llm_triage_penalty: float = 0.5

    # --- paths (overridable for tests) ---
    ha_config_dir: str = "/homeassistant"
    state_dir: str = "/config"

    def __post_init__(self) -> None:
        merged_domains = set(DEFAULT_EXCLUDED_DOMAINS) | {
            d.strip().lower() for d in self.excluded_domains if d and d.strip()
        }
        self.excluded_domains = sorted(merged_domains)
        merged_entities = set(DEFAULT_EXCLUDED_ENTITIES) | {
            e.strip().lower() for e in self.excluded_entities if e and e.strip()
        }
        self.excluded_entities = sorted(merged_entities)
        self.excluded_users = [u for u in self.excluded_users if u]
        self.min_consistency = min(max(float(self.min_consistency), 0.0), 1.0)
        self.min_confidence = min(max(float(self.min_confidence), 0.0), 1.0)
        self.min_support = min(max(float(self.min_support), 0.0), 1.0)
        self.backtest_min_precision = min(max(float(self.backtest_min_precision), 0.0), 1.0)
        self.min_occurrences = max(int(self.min_occurrences), 2)
        self.llm_triage_penalty = min(max(float(self.llm_triage_penalty), 0.0), 1.0)
        self.llm_classification_batch = max(int(self.llm_classification_batch), 5)
        self.override_window_seconds = max(int(self.override_window_seconds), 1)

    # ------------------------------------------------------------------
    def is_excluded(self, entity_id: str) -> bool:
        """Return True when *entity_id* must not take part in mining.

        Excluded entities may be given as plain entity ids or as glob patterns
        (``sensor.*_uptime``).
        """
        if not entity_id or "." not in entity_id:
            return True
        entity_id = entity_id.lower()
        domain = entity_id.split(".", 1)[0]
        if domain in self.excluded_domains:
            return True
        for pattern in self.excluded_entities:
            if pattern == entity_id or fnmatch(entity_id, pattern):
                return True
        return False

    @property
    def llm_enabled(self) -> bool:
        """True when a provider is configured at all."""
        return (self.llm_provider or "none").lower() not in ("", "none", "off", "disabled")

    @property
    def ai_features_requested(self) -> dict[str, bool]:
        """The optional AI features the user asked for, on or off."""
        return {
            "entity_classification": bool(self.llm_entity_classification),
            "hypotheses": bool(self.llm_hypotheses),
            "triage": bool(self.llm_triage),
        }

    @property
    def any_ai_feature(self) -> bool:
        return any(self.ai_features_requested.values())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------
    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> Options:
        """Build options from a raw mapping, ignoring unknown keys."""
        data = dict(data or {})
        known = {f.name: f for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            spec = known.get(key)
            if spec is None or value is None:
                continue
            if spec.type in ("bool", bool):
                kwargs[key] = _as_bool(value, bool(spec.default))
            elif spec.type in ("int", int):
                try:
                    kwargs[key] = int(value)
                except (TypeError, ValueError):
                    _LOGGER.warning("Ignoring non-integer option %s=%r", key, value)
            elif spec.type in ("float", float):
                try:
                    kwargs[key] = float(value)
                except (TypeError, ValueError):
                    _LOGGER.warning("Ignoring non-float option %s=%r", key, value)
            elif spec.type in ("list[str]",):
                if isinstance(value, str):
                    kwargs[key] = [v.strip() for v in value.split(",") if v.strip()]
                elif isinstance(value, (list, tuple)):
                    kwargs[key] = [str(v) for v in value]
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> Options:
        """Load options from the Supervisor-provided ``options.json``.

        Missing or unreadable files fall back to the built-in defaults, which is
        the documented zero-config path.
        """
        raw: dict[str, Any] = {}
        candidate = Path(path or os.environ.get("AMMINER_OPTIONS_PATH", "/data/options.json"))
        if candidate.is_file():
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8")) or {}
            except (OSError, json.JSONDecodeError) as err:
                _LOGGER.warning("Could not read %s (%s); using defaults", candidate, err)
        else:
            _LOGGER.info("No options file at %s; using defaults", candidate)

        for env_key, opt_key in (
            ("AMMINER_LOG_LEVEL", "log_level"),
            ("AMMINER_HA_CONFIG_DIR", "ha_config_dir"),
            ("AMMINER_STATE_DIR", "state_dir"),
        ):
            if os.environ.get(env_key):
                raw[opt_key] = os.environ[env_key]

        return cls.from_mapping(raw)
