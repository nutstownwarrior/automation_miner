"""The analysis pipeline: discover -> mine -> enrich -> backtest -> check -> store.

Everything degrades.  A missing recorder, an unreadable registry, no external
signals, no LLM - each removes capability but never stops the run.  Whatever the
pipeline could not do is reported in ``RunReport.degradations`` and shown in the
UI, so the user always knows *why* they are seeing fewer suggestions.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from . import conflicts as conflict_checks
from . import gaps as gap_analysis
from .automations import load_existing_automations
from .backtest import backtest_all
from .config import Options
from .discovery.ha_config import HAConfig
from .discovery.recorder import Recorder, open_recorder
from .enrich.detect import detect_signals
from .enrich.signals import build_signal_store
from .entities import build_resolver
from .ha_api import HAClient
from .miners import association, conditional, energy, motif, sequence, stale, time_of_day
from .miners.base import Candidate
from .recorderdb import causality
from .recorderdb.queries import ORIGIN_EVENT_TYPES, RecorderQueries
from .store import Store

_LOGGER = logging.getLogger(__name__)

#: Below this much raw history, sequence/association mining is not trustworthy.
MIN_DAYS_FOR_SEQUENCE_MINING = 7


@dataclass
class RunReport:
    """Everything one analysis run learned, for the UI and the run log."""

    started_ts: float = field(default_factory=time.time)
    finished_ts: float | None = None
    status: str = "running"
    error: str | None = None
    window: tuple[float, float] = (0.0, 0.0)
    window_days: float = 0.0
    recorder: dict[str, Any] = field(default_factory=dict)
    entities: dict[str, Any] = field(default_factory=dict)
    signals: dict[str, list[str]] = field(default_factory=dict)
    causality: dict[str, int] = field(default_factory=dict)
    overrides: int = 0
    miner_counts: dict[str, int] = field(default_factory=dict)
    surfaced: int = 0
    rejected: int = 0
    conflicted: int = 0
    gaps: int = 0
    degradations: list[str] = field(default_factory=list)
    state_rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_ts": self.started_ts,
            "finished_ts": self.finished_ts,
            "status": self.status,
            "error": self.error,
            "window_start_ts": self.window[0],
            "window_end_ts": self.window[1],
            "window_days": round(self.window_days, 1),
            "recorder": self.recorder,
            "entities": self.entities,
            "signals": self.signals,
            "causality": self.causality,
            "overrides": self.overrides,
            "miner_counts": self.miner_counts,
            "surfaced": self.surfaced,
            "rejected": self.rejected,
            "conflicted": self.conflicted,
            "gaps": self.gaps,
            "degradations": self.degradations,
            "state_rows": self.state_rows,
        }


def resolve_window(options: Options, recorder: Recorder) -> tuple[float, float, list[str]]:
    """Pick the analysis window: ``auto`` == min(available history, 60 days)."""
    notes: list[str] = []
    now = time.time()
    info = recorder.info
    newest = info.newest_state_ts or now
    oldest = info.oldest_state_ts or (newest - 86400.0)
    available_days = max((newest - oldest) / 86400.0, 0.0)

    if options.analysis_window_days > 0:
        requested = float(options.analysis_window_days)
        if requested > available_days:
            notes.append(
                f"Requested a {requested:.0f}-day window but only {available_days:.1f} days of "
                "history exist; using what is available."
            )
        days = min(requested, max(available_days, 1.0))
    else:
        days = min(available_days, float(options.max_auto_window_days))
        days = max(days, 1.0)
        notes.append(
            f"Analysis window auto-selected: {days:.1f} days "
            f"(min of available history and {options.max_auto_window_days} days)."
        )
    start_ts = max(oldest, newest - days * 86400.0)
    return start_ts, newest, notes


def run_analysis(
    options: Options,
    store: Store,
    client: HAClient | None = None,
    ha_config: HAConfig | None = None,
) -> tuple[RunReport, list[Candidate]]:
    """Run one complete analysis and persist the results."""
    report = RunReport()
    run_id = store.start_run()
    candidates: list[Candidate] = []

    try:
        ha_config = ha_config or HAConfig(options.ha_config_dir)
        client = client if client is not None else HAClient()
        if not ha_config.available:
            report.degradations.append(
                f"configuration.yaml not readable at {options.ha_config_dir}; "
                "using defaults for recorder discovery."
            )

        # --- entity resolution ---------------------------------------
        resolver = build_resolver(options.ha_config_dir, client)
        report.entities = resolver.stats()
        if not resolver.entities:
            report.degradations.append(
                "No entities could be resolved from the registry, the WebSocket API or "
                "/api/states. Suggestions will be unnamed and no validation is possible."
            )
        elif not resolver.sources.get("registry"):
            report.degradations.append(
                "Entity registry unavailable; areas, devices and labels are missing from "
                "suggestions (fell back to /api/states)."
            )
        elif not resolver.sources.get("states"):
            report.degradations.append(
                "Home Assistant's /api/states could not be read; entities without a unique_id "
                "(template/YAML sensors) are invisible to this run."
            )

        # --- recorder -------------------------------------------------
        recorder = open_recorder(ha_config, options)
        report.recorder = recorder.info.as_dict()
        if not recorder.available:
            report.degradations.append(
                f"Recorder unavailable ({recorder.info.error}); no history mining is possible."
            )
            report.status = "degraded"
            report.finished_ts = time.time()
            store.finish_run(run_id, "degraded", report.as_dict())
            return report, []

        start_ts, end_ts, window_notes = resolve_window(options, recorder)
        report.window = (start_ts, end_ts)
        report.window_days = (end_ts - start_ts) / 86400.0
        report.degradations.extend(window_notes)

        queries = RecorderQueries(recorder.engine, recorder.info)
        if not resolver.entities:
            # No registry and no /api/states: the recorder is the only remaining
            # source of the entity set, so mine with names alone.
            resolver.seed_from_recorder(queries.entity_ids())
            report.entities = resolver.stats()

        changes = queries.state_changes(start_ts, end_ts, with_attributes=True)
        events = queries.events(start_ts, end_ts, ORIGIN_EVENT_TYPES)
        report.state_rows = len(changes)
        _LOGGER.info("Loaded %d state rows and %d events", len(changes), len(events))

        # --- causality & overrides -----------------------------------
        changes, index = causality.annotate(changes, events, options.excluded_users)
        report.causality = causality.causality_stats(changes)
        if not index.has_user_context:
            report.degradations.append(
                "No context_user_id found in this history window: human and automated changes "
                "cannot be told apart reliably, so all mining runs at reduced confidence."
            )
        overrides = causality.detect_overrides(changes, options.override_window_seconds)
        report.overrides = len(overrides)
        store.record_overrides(overrides)
        override_counts = store.override_counts()

        # --- enrichment ----------------------------------------------
        signals = detect_signals(resolver)
        report.signals = signals.as_dict()
        if not signals.present():
            report.degradations.append(
                "No external signals (sun, weather, presence, price) detected; only intrinsic "
                "patterns are mined."
            )
        signal_store = build_signal_store(
            changes, signals.all_entities, queries, (start_ts, end_ts)
        )
        # Every entity a candidate might reference must be simulatable, so the
        # backtester gets a store covering the whole mined entity set.
        all_entities = sorted({c.entity_id for c in changes} | set(signals.all_entities))
        full_store = build_signal_store(changes, all_entities, queries, (start_ts, end_ts))

        # --- mining ---------------------------------------------------
        window = (start_ts, end_ts)
        produced: dict[str, list[Candidate]] = {}

        produced["time_of_day"] = time_of_day.mine(changes, options, window, resolver)
        produced["conditional"] = conditional.mine(
            changes, options, signals, signal_store, window, resolver
        )
        produced["motif"] = motif.mine(changes, options, full_store, window, resolver)
        produced["energy_shift"] = energy.mine(
            changes, options, signals, signal_store, window, resolver
        )

        if report.window_days >= MIN_DAYS_FOR_SEQUENCE_MINING:
            produced["association"] = association.mine(changes, options, window, resolver)
            produced["sequence"] = sequence.mine(changes, options, window, resolver)
        else:
            produced["association"] = []
            produced["sequence"] = []
            report.degradations.append(
                f"Only {report.window_days:.1f} days of history (< "
                f"{MIN_DAYS_FOR_SEQUENCE_MINING}): association and sequence mining are disabled. "
                "Switch the recorder to MariaDB and raise purge_keep_days to enable them."
            )

        audit_findings = stale.mine(changes, resolver, options, window, override_counts)
        report.miner_counts = {name: len(items) for name, items in produced.items()}
        report.miner_counts["audit"] = len(audit_findings)

        mined: list[Candidate] = [c for items in produced.values() for c in items]

        # --- dismissals ------------------------------------------------
        dismissed = store.dismissed_ids() | store.dismissed_signatures()
        before = len(mined)
        mined = [c for c in mined if c.id not in dismissed]
        if before != len(mined):
            _LOGGER.info("Filtered %d previously dismissed candidates", before - len(mined))

        # --- backtest --------------------------------------------------
        passed, rejected = backtest_all(
            mined, changes, full_store, options, window, overrides
        )
        report.surfaced = len(passed)
        report.rejected = len(rejected)

        # --- conflicts -------------------------------------------------
        existing = load_existing_automations(ha_config, resolver)
        conflict_checks.annotate_candidates(passed, existing, resolver)
        report.conflicted = sum(
            1 for c in passed if conflict_checks.has_blocking_conflict(c)
        )

        # --- persist ---------------------------------------------------
        for candidate in passed:
            store.upsert_suggestion(
                candidate.id,
                candidate.miner,
                candidate.title,
                candidate.describe(resolver),
                candidate.score,
                candidate.as_dict(resolver),
                run_id,
            )
            if candidate.backtest:
                store.save_backtest(candidate.id, candidate.backtest)
        for candidate in audit_findings:
            if candidate.id in dismissed:
                continue
            store.upsert_suggestion(
                candidate.id,
                candidate.miner,
                candidate.title,
                candidate.description,
                candidate.score,
                candidate.as_dict(resolver),
                run_id,
            )
        store.prune_suggestions(run_id)

        # --- gaps ------------------------------------------------------
        gap_suggestions = gap_analysis.suggest(
            resolver, signals, changes, passed, recorder.info
        )
        report.gaps = len(gap_suggestions)
        for gap in gap_suggestions:
            store.upsert_gap(gap.id, gap.kind, gap.title, gap.as_dict())

        store.set_meta("last_audit", str(int(time.time())))
        store.set_meta(
            "existing_automation_audit",
            __import__("json").dumps(conflict_checks.audit_existing(existing, resolver)),
        )

        report.status = "ok"
        candidates = passed + audit_findings
        recorder.close()

    except Exception as err:  # noqa: BLE001 - a failed run must never kill the add-on
        report.status = "error"
        report.error = f"{type(err).__name__}: {err}"
        _LOGGER.error("Analysis run failed: %s\n%s", err, traceback.format_exc())
        report.finished_ts = time.time()
        store.finish_run(run_id, "error", report.as_dict(), report.error)
        return report, []

    report.finished_ts = time.time()
    store.finish_run(run_id, report.status, report.as_dict())
    _LOGGER.info(
        "Run finished in %.1fs: %d surfaced, %d rejected, %d gaps",
        report.finished_ts - report.started_ts,
        report.surfaced,
        report.rejected,
        report.gaps,
    )
    return report, candidates
