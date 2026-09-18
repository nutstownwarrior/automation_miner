"""The analysis pipeline: discover -> mine -> enrich -> backtest -> check -> store.

Everything degrades.  A missing recorder, an unreadable registry, no external
signals, no LLM - each removes capability but never stops the run.  Whatever the
pipeline could not do is reported in ``RunReport.degradations`` and shown in the
UI, so the user always knows *why* they are seeing fewer suggestions.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from . import backtest as backtest_module
from . import conflicts as conflict_checks
from . import gaps as gap_analysis
from . import notify as notifier
from .automations import load_existing_automations
from .backtest import backtest_all
from .config import Options
from .discovery.ha_config import HAConfig
from .discovery.recorder import Recorder, open_recorder
from .enrich.detect import detect_signals
from .enrich.signals import SignalStore, build_signal_store
from .entities import build_resolver
from .ha_api import HAClient
from .learn import ranking as ranking_module
from .llm import areas as llm_areas
from .llm import audit as llm_audit
from .llm import classify as llm_classify
from .llm import explain as llm_explain
from .llm import gaps as llm_gap_proposals
from .llm import hypothesis as llm_hypothesis
from .llm import preferences as llm_preferences
from .llm import scenes as llm_scenes
from .llm import triage as llm_triage
from .llm.provider import build_provider
from .miners import association, conditional, energy, motif, sequence, stale, time_of_day
from .miners.base import Candidate
from .recorderdb import causality
from .recorderdb.models import StateChange
from .recorderdb.queries import ORIGIN_EVENT_TYPES, RecorderQueries
from .store import STATUS_SHADOW, Store

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
    #: miner name -> the error that stopped it, for miners that failed
    miner_errors: dict[str, str] = field(default_factory=dict)
    #: AI feature name -> the error that stopped it
    ai_errors: dict[str, str] = field(default_factory=dict)
    #: Per optional AI feature: whether it ran and what it contributed.
    ai: dict[str, Any] = field(default_factory=dict)
    surfaced: int = 0
    rejected: int = 0
    conflicted: int = 0
    gaps: int = 0
    #: What was announced to Home Assistant about this run's new suggestions.
    notified: dict[str, Any] = field(default_factory=dict)
    #: Would-be fires recorded for suggestions the user asked to shadow-test.
    shadow_fires: int = 0
    #: Suggestions hidden because they match a preference learned from the
    #: reasons the user gave when dismissing things before.
    suppressed: int = 0
    #: The acceptance-ranking model's own stats (n_labels, whether it fell
    #: back to the prior and why), never including the weights themselves -
    #: for the Status page, not for reconstructing the model.
    ranking: dict[str, Any] = field(default_factory=dict)
    #: Post-deployment health of automations this add-on has applied (see
    #: amminer.health) - how many were checked and their verdicts, never the
    #: full detail (that lives in the store, for the automations page).
    automations: dict[str, Any] = field(default_factory=dict)
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
            "miner_errors": self.miner_errors,
            "ai_errors": self.ai_errors,
            "ai": self.ai,
            "surfaced": self.surfaced,
            "rejected": self.rejected,
            "conflicted": self.conflicted,
            "gaps": self.gaps,
            "notified": self.notified,
            "shadow_fires": self.shadow_fires,
            "suppressed": self.suppressed,
            "ranking": self.ranking,
            "automations": self.automations,
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


def _log_shadow_fires(
    store: Store,
    changes: Sequence[StateChange],
    signal_store: SignalStore,
    options: Options,
    window: tuple[float, float],
) -> int:
    """Replay every shadowed suggestion over history it has not been scored on."""
    # Imported here: runner imports this module, so importing it back at module
    # scope would be a cycle.
    from .runner import candidate_from_payload

    logged = 0
    for row in store.list_suggestions(status=STATUS_SHADOW):
        payload = row.get("payload") or {}
        if not payload.get("actions"):
            continue
        candidate = candidate_from_payload(payload)
        since = store.last_shadow_ts(row["id"])
        for ts, matched in backtest_module.shadow_evaluate(
            candidate, changes, signal_store, options, window, since_ts=since
        ):
            store.log_shadow_fire(row["id"], ts, matched=matched)
            logged += 1
    return logged


def run_analysis(
    options: Options,
    store: Store,
    client: HAClient | None = None,
    ha_config: HAConfig | None = None,
    resolver=None,
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
        # The caller may already have built one; reading the registry files and
        # calling /api/states a second time doubles this run's load on Core for
        # an object we then discard - and left Runner.resolver, which the UI and
        # every preview use, as a different object from the one that was mined
        # and backtested against.
        if resolver is None:
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
        unknown = report.causality.get("unknown", 0)
        if unknown and len(changes):
            share = unknown / len(changes)
            if share >= 0.1:
                report.degradations.append(
                    f"{share:.0%} of state rows carry no context at all, so who caused them "
                    "is genuinely unknown. Those rows take no part in mining rather than "
                    "being counted as device activity."
                )
        overrides = causality.detect_overrides(changes, options.override_window_seconds)
        report.overrides = len(overrides)
        store.record_overrides(overrides)
        override_counts = store.override_counts()

        # --- optional AI assistance ----------------------------------
        # Every feature here is off by default and additive: with them disabled
        # the run is bit-for-bit what it was before they existed.
        provider = None
        if options.any_ai_feature:
            if not options.llm_enabled:
                report.degradations.append(
                    "AI assistance is switched on but llm_provider is 'none', so the "
                    "assisted features did nothing. Set a provider to use them."
                )
                report.ai = {
                    name: {"requested": True, "ran": False, "reason": "no llm_provider"}
                    for name, on in options.ai_features_requested.items()
                    if on
                }
            else:
                provider = build_provider(options)
                status = provider.status()
                if not status.available:
                    report.degradations.append(
                        f"AI assistance is switched on but the {status.provider} provider is "
                        f"not usable ({status.error}); the assisted features were skipped."
                    )
                    report.ai = {
                        name: {"requested": True, "ran": False, "reason": status.error}
                        for name, on in options.ai_features_requested.items()
                        if on
                    }
                    provider = None

        def run_ai(name: str, func, *args, **kwargs):
            """Run one AI feature in isolation, like a miner."""
            try:
                outcome = func(*args, **kwargs)
            except Exception as err:  # noqa: BLE001 - assistance is never fatal
                _LOGGER.exception("AI feature %s failed: %s", name, err)
                report.ai[name] = {
                    "requested": True,
                    "ran": False,
                    "reason": f"{type(err).__name__}: {err}",
                }
                report.degradations.append(
                    f"The AI {name.replace('_', ' ')} step failed and was skipped "
                    f"({type(err).__name__}: {err})."
                )
                # "partial" means something the user asked for did not happen.
                # A failed AI feature is exactly that, and reporting it as a
                # plain "ok" run made the word mean only "a miner failed".
                report.ai_errors[name] = f"{type(err).__name__}: {err}"
                return None
            report.ai[name] = {"requested": True, "ran": True, **outcome.as_dict()}
            return outcome

        def run_stage(name, func, *args):
            """Run one post-mining stage in isolation.

            Same contract as run_miner, for the parts of the run that come
            after it: a failure costs that stage and is reported, rather than
            discarding work that already succeeded.  ``None`` means it failed.
            """
            try:
                return func(*args)
            except Exception as err:  # noqa: BLE001 - one stage must not end the run
                _LOGGER.exception("Stage %s failed: %s", name, err)
                report.miner_errors[name] = f"{type(err).__name__}: {err}"
                report.degradations.append(
                    f"The {name} step failed ({type(err).__name__}: {err})."
                )
                return None

        # --- enrichment ----------------------------------------------
        # Areas first: every description, prompt and card built below reads
        # `area_name`, so an inference made afterwards would be invisible to
        # everything that could have used it.
        if provider is not None and options.llm_areas:
            inferred = run_ai("area_inference", llm_areas.infer, resolver, provider)
            if inferred is not None:
                # Isolated like the model call itself: applying the result
                # mutates the resolver every later step reads, and an exception
                # here used to fall through to the outer handler - which ends
                # the whole run before a single miner has been asked anything.
                run_stage("applying area inferences", llm_areas.apply_inferences,
                          resolver, inferred)
                report.ai["area_inference"].update(inferred.as_dict())

        signals = detect_signals(resolver)

        if provider is not None and options.llm_entity_classification:
            classification = run_ai(
                "entity_classification",
                llm_classify.classify,
                resolver,
                provider,
                options,
                store,
                options.llm_classification_batch,
            )
            if classification is not None:
                llm_classify.apply_to_signals(signals, classification)
                report.ai["entity_classification"].update(classification.as_dict())

        report.signals = signals.as_dict()
        if not signals.present():
            report.degradations.append(
                "No external signals (sun, weather, presence, price) detected; only intrinsic "
                "patterns are mined."
            )
        # One store, not two.  Every entity a candidate might reference has to
        # be simulatable, so this covers the whole mined entity set - which is
        # a superset of the detected signals.  Building a signals-only store as
        # well meant a second full pass over the same state history for a
        # strict subset of the same data, and both kept alive for the rest of
        # the run.  The miners that used to take the smaller one look entities
        # up by id and never enumerate it, so they see exactly what they did.
        all_entities = sorted({c.entity_id for c in changes} | set(signals.all_entities))
        window = (start_ts, end_ts)
        # The full window, used below for the backtest step (which needs to
        # see the holdout to judge candidates against it) and for the audit
        # miner (see the note by that call): unlike everything mined for a
        # suggestion, staleness is a fact about age and configuration, not a
        # statistical pattern being generalised, so there is nothing for it to
        # leak by seeing the whole window.
        full_store = build_signal_store(changes, all_entities, queries, window)

        # --- train/holdout split ---------------------------------------
        # The final slice of the window is never mined from - only used, in
        # the backtest step below, to judge what was mined from the rest.  A
        # candidate a miner never saw the holdout to build cannot have been
        # shaped by it, which is the whole point: precision measured there is
        # a genuine estimate of how the rule does on history it has not seen,
        # not a report card on the exam it was allowed to study from.
        split = backtest_module.split_window(window, options)
        train_window = split.train
        train_changes = [c for c in changes if c.ts < train_window[1]]
        # Its own store, built from training rows alone, so a conditional
        # miner cannot read a signal's holdout-period values either - a
        # temperature threshold picked with next week's readings in view would
        # leak exactly the same way a trigger time picked from them would.
        train_store = build_signal_store(train_changes, all_entities, queries, train_window)
        if (
            split.holdout_days < options.backtest_min_holdout_days
            or split.train_days < options.backtest_min_train_days
        ):
            report.degradations.append(
                f"Only {split.holdout_days:.1f} days of history can be held out to validate "
                f"suggestions against (need at least {options.backtest_min_holdout_days}), "
                f"leaving {split.train_days:.1f} to mine from (need at least "
                f"{options.backtest_min_train_days}). Suggestions are graded against the "
                "whole analysis window instead of history they were not mined from, until "
                "there is more of it."
            )

        # --- mining ---------------------------------------------------
        produced: dict[str, list[Candidate]] = {}

        def run_miner(name, func, *args):
            """Run one miner in isolation.

            A miner that raises must cost only its own findings - a dependency
            with a changed signature, or one pathological entity, must never
            take down a run that every other miner could have contributed to.
            """
            try:
                return func(*args)
            except Exception as err:  # noqa: BLE001 - one miner must not end the run
                _LOGGER.exception("Miner %s failed: %s", name, err)
                report.miner_errors[name] = f"{type(err).__name__}: {err}"
                report.degradations.append(
                    f"The {name.replace('_', ' ')} miner failed and was skipped "
                    f"({type(err).__name__}: {err}). Other miners still ran."
                )
                return []

        produced["time_of_day"] = run_miner(
            "time_of_day", time_of_day.mine, train_changes, options, train_window, resolver
        )
        produced["conditional"] = run_miner(
            "conditional", conditional.mine, train_changes, options, signals, train_store,
            train_window, resolver,
        )
        produced["motif"] = run_miner(
            "motif", motif.mine, train_changes, options, train_store, train_window, resolver
        )
        produced["energy_shift"] = run_miner(
            "energy_shift", energy.mine, train_changes, options, signals, train_store,
            train_window, resolver,
        )

        if split.train_days >= MIN_DAYS_FOR_SEQUENCE_MINING:
            produced["association"] = run_miner(
                "association", association.mine, train_changes, options, train_window, resolver
            )
            produced["sequence"] = run_miner(
                "sequence", sequence.mine, train_changes, options, train_window, resolver
            )
        else:
            produced["association"] = []
            produced["sequence"] = []
            report.degradations.append(
                f"Only {split.train_days:.1f} days of training history (< "
                f"{MIN_DAYS_FOR_SEQUENCE_MINING}): association and sequence mining are disabled. "
                "Switch the recorder to MariaDB and raise purge_keep_days to enable them."
            )

        # Deliberately the full window, not train_changes/train_window: a stale
        # or unused automation is found by its age and configuration
        # (last_triggered, whether anything still references it), not by
        # mining a behavioural pattern that then has to generalise.  There is
        # no train/holdout split to leak across here.
        audit_findings = run_miner(
            "audit", stale.mine, changes, resolver, options, window, override_counts
        )
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
        # Everything from here on is isolated the same way the miners are.  It
        # used to fall through to the outer handler, which sets status=error
        # and returns nothing - so an exception in any one of backtesting,
        # conflict checking or gap analysis threw away every candidate every
        # miner had already produced, and persisted none of them.
        backtested = run_stage(
            "backtest", backtest_all, mined, changes, full_store, options, window, overrides
        )
        if backtested is None:
            # Without a backtest nothing may be surfaced: an unmeasured rule is
            # exactly what this project exists not to suggest.
            passed, rejected = [], list(mined)
        else:
            passed, rejected = backtested
            # holdout validation is per-candidate: the window can be long
            # enough for a trustworthy split and a particular candidate can
            # still have had nothing happen in it either way (see
            # BacktestResult.holdout_reason).  That is worth a run-level note
            # too, distinct from "the window itself was too short" above.
            if any(
                (c.backtest or {}).get("holdout_reason") == "no_activity_in_holdout"
                for c in passed + rejected
            ):
                report.degradations.append(
                    "Some suggestions had no activity in the held-out period to check them "
                    "against, so they were graded on the whole analysis window instead."
                )
        # --- AI hypotheses: propose, then measure with the same gate ---
        if provider is not None and options.llm_hypotheses and rejected:
            hypotheses = run_ai(
                "hypotheses",
                llm_hypothesis.propose_and_verify,
                rejected,
                changes,
                full_store,
                options,
                window,
                provider,
                resolver,
                overrides,
                train_store,
            )
            if hypotheses is not None and hypotheses.accepted:
                accepted_origins = {
                    c.extra.get("hypothesis", {}).get("origin_candidate")
                    for c in hypotheses.accepted
                }
                passed.extend(hypotheses.accepted)
                # A rejected candidate that a verified condition rescued is no
                # longer a rejection; it was superseded, not discarded.
                rejected = [c for c in rejected if c.id not in accepted_origins]

        # --- AI triage: advisory demotion only -------------------------
        if provider is not None and options.llm_triage and passed:
            verdicts = run_ai("triage", llm_triage.triage, passed, provider, resolver)
            if verdicts is not None:
                llm_triage.apply_verdicts(passed, verdicts, options.llm_triage_penalty)
                report.ai["triage"].update(verdicts.as_dict())
                passed.sort(key=lambda c: c.score, reverse=True)

        # --- AI scenes: propose a grouping, measure it as one rule -----
        if provider is not None and options.llm_scenes and len(passed) >= 2:
            grouped = run_ai(
                "scenes",
                llm_scenes.propose_and_verify,
                passed,
                changes,
                full_store,
                options,
                window,
                provider,
                resolver,
                overrides,
            )
            if grouped is not None and grouped.accepted:
                # Added alongside its members, never instead of them: a scene is
                # a proposal about several suggestions the user has already been
                # shown, and it may not make any of them disappear.
                run_stage("applying scenes", llm_scenes.apply_scenes, passed, grouped)
                # A scene's id is a hash of its parts, so a dismissed grouping
                # comes back identical every run.  Mined candidates are filtered
                # against the dismissal list before backtesting; scenes are built
                # afterwards, so they have to be filtered here or a dismissed
                # scene keeps being counted as surfaced forever.
                passed.extend(c for c in grouped.accepted if c.id not in dismissed)
                passed.sort(key=lambda c: c.score, reverse=True)

        # --- AI explanations: rendering only ---------------------------
        if provider is not None and options.llm_explain and passed:
            explained = run_ai("explanations", llm_explain.explain, passed, provider, resolver)
            if explained is not None:
                run_stage("applying explanations", llm_explain.apply_explanations,
                          passed, explained)

        # --- AI preferences: what the user has already said no to ------
        suppressed_ids: set[str] = set()
        if provider is not None and options.llm_preferences:
            def _learn() -> tuple[list, str | None]:
                found, why = llm_preferences.learn(
                    store.dismissals_with_reasons(), provider
                )
                # Only a successful call may rewrite the stored set.  `learn`
                # reports a provider that timed out the same way it reports
                # "nothing to generalise" - an empty list - and
                # `save_preferences` reads an empty list as "the model withdrew
                # every preference".  Saving unconditionally therefore deleted
                # every learned preference the user had, silently, on one bad
                # night.
                if why is None:
                    store.save_preferences([p.as_dict() for p in found])
                # Read back rather than using what the model just said: the
                # stored row is the one the user can switch off, rewrite or
                # delete, and it is that text - not the model's - that decides
                # what gets hidden.  Preferences the user wrote by hand are in
                # here too, and apply whether or not anything was learned.
                return [
                    llm_preferences.Preference.from_row(row)
                    for row in store.list_preferences(active_only=True)
                ], why

            # Learning runs whether or not anything survived the backtest: the
            # preference set is derived from dismissals, not from this run's
            # candidates, and a quiet night should not stop it being refreshed.
            usable, learn_error = run_stage("learning preferences", _learn) or ([], None)
            matched = run_ai(
                "preferences",
                llm_preferences.apply_preferences,
                passed,
                usable,
                provider,
                resolver,
            ) if passed else None
            if learn_error and "preferences" in report.ai:
                report.ai["preferences"]["learn_error"] = learn_error
                report.degradations.append(
                    "Standing preferences could not be refreshed this run "
                    f"({learn_error}); the ones you already have were kept."
                )
            if matched is not None:
                suppressed_ids = set(matched.suppressed)
                for candidate in passed:
                    hidden = matched.suppressed.get(candidate.id)
                    if hidden:
                        candidate.extra["suppressed_by"] = dict(hidden)
        report.suppressed = len(suppressed_ids)

        report.surfaced = len(passed) - len(suppressed_ids)
        report.rejected = len(rejected)

        # --- shadow mode -----------------------------------------------
        # "Shadow-test only" set a status and logged nothing, so the detail
        # page reported 0 would-be fires forever.  Every run now replays the
        # rules the user asked to watch against the history since it last
        # looked, and records each fire.
        report.shadow_fires = run_miner(
            "shadow", _log_shadow_fires, store, changes, full_store, options, window
        ) or 0

        # --- conflicts -------------------------------------------------
        existing = run_stage("existing automations", load_existing_automations,
                             ha_config, resolver) or []

        def _annotate_conflicts() -> int:
            conflict_checks.annotate_candidates(passed, existing, resolver)
            return sum(1 for c in passed if conflict_checks.has_blocking_conflict(c))

        conflicted = run_stage("conflicts", _annotate_conflicts)
        if conflicted is None:
            # Unchecked is not the same as clean, and the user is told so.
            report.degradations.append(
                "Conflict checking failed, so these suggestions have not been compared "
                "against the automations you already have."
            )
        report.conflicted = conflicted or 0

        # --- automation health: does what we already shipped still perform? -
        # Reuses exactly the events/full_store/overrides this run already
        # loaded for causality classification and backtesting - see
        # amminer.health's own docstring for why nothing here re-queries the
        # recorder.  `existing` is the same list conflict checking just used,
        # keyed the same way.
        def _check_health() -> list[dict[str, Any]]:
            from . import health as health_module

            results = health_module.evaluate_all(
                store, existing, events, full_store, overrides, options, window
            )
            return [r.as_dict() for r in results]

        health_results = run_stage("automation health", _check_health)
        if health_results is None:
            report.degradations.append(
                "Could not check the health of automations already applied; last "
                "known status is shown instead."
            )
        else:
            by_verdict: dict[str, int] = {}
            for entry in health_results:
                by_verdict[entry["verdict"]] = by_verdict.get(entry["verdict"], 0) + 1
            report.automations = {"checked": len(health_results), "by_verdict": by_verdict}

        # --- ranking: a calibrated ordering, never a gate ---------------
        # Retrained from scratch every run from the user's own accept/dismiss
        # history so far - amminer/learn/ranking.py documents why that is
        # cheap, why it is safe with zero history (a hand-set prior), and why
        # a noisy handful of decisions cannot make the ordering worse than
        # that prior (a cross-validated guard falls back to it otherwise).
        #
        # This runs after backtesting and conflict checking and is only ever
        # given `passed` - it has no way to rescue a candidate `backtest_all`
        # rejected or to hide one that has a blocking conflict, because it
        # never sees `rejected` and never touches `conflicts` or the
        # `passed`/`rejected` split itself, only `candidate.extra`.
        if options.ranking_enabled and passed:
            def _rank() -> dict[str, Any]:
                examples = ranking_module.labels_from_rows(store.ranking_labels())
                model = ranking_module.train_from_labels(examples, l2=options.ranking_prior_strength)
                store.save_ranking_model(model.as_dict())
                seen_counts = store.seen_counts([c.id for c in passed])
                ranking_module.rank_candidates(passed, model, seen_counts)
                return {k: v for k, v in model.as_dict().items() if k != "weights"}

            ranking_summary = run_stage("ranking", _rank)
            if ranking_summary is None:
                report.degradations.append(
                    "Learning your acceptance patterns failed this run; suggestions are "
                    "ordered by each miner's own score instead."
                )
            else:
                report.ranking = ranking_summary

        # --- persist ---------------------------------------------------
        # Per candidate, not per loop: one candidate whose payload will not
        # serialise must not take the other forty with it.
        for candidate in passed:
            def _save(candidate=candidate) -> bool:
                ranking_info = candidate.extra.get("ranking") or {}
                store.upsert_suggestion(
                    candidate.id,
                    candidate.miner,
                    candidate.title,
                    candidate.describe(resolver),
                    candidate.score,
                    candidate.as_dict(resolver),
                    run_id,
                    accept_probability=ranking_info.get("probability"),
                )
                if candidate.backtest:
                    store.save_backtest(candidate.id, candidate.backtest)
                # Only a brand-new suggestion may be hidden, and only while the
                # preference that hid it is still switched on.  Both conditions
                # are checked inside the write rather than out here: this run
                # decided what to hide from a snapshot taken minutes ago, and
                # the user may have accepted the rule or switched the
                # preference off since.
                hidden_by = candidate.extra.get("suppressed_by") or {}
                if hidden_by.get("preference"):
                    store.suppress_if_active(candidate.id, hidden_by["preference"])
                return True

            run_stage(f"saving {candidate.id}", _save)
        for candidate in audit_findings:
            if candidate.id in dismissed:
                continue

            def _save_audit(candidate=candidate) -> bool:
                store.upsert_suggestion(
                    candidate.id,
                    candidate.miner,
                    candidate.title,
                    candidate.description,
                    candidate.score,
                    candidate.as_dict(resolver),
                    run_id,
                )
                return True

            run_stage(f"saving {candidate.id}", _save_audit)
        run_stage("pruning", store.prune_suggestions, run_id)

        # --- gaps ------------------------------------------------------
        def _suggest_gaps() -> int:
            gap_suggestions = gap_analysis.suggest(
                resolver, signals, changes, passed, recorder.info
            )
            # Additive only, and after the detector: a proposal from world
            # knowledge is a weaker signal than a detected gap, and can never
            # replace, reword or reorder one.
            if provider is not None and options.llm_gaps:
                proposals = run_ai(
                    "gap_proposals",
                    llm_gap_proposals.propose,
                    signals,
                    gap_analysis.human_actions_by_domain(changes),
                    gap_suggestions,
                    provider,
                    set(resolver.known_entity_ids()),
                )
                if proposals is not None:
                    gap_suggestions = list(gap_suggestions) + list(proposals.accepted)
                    report.ai["gap_proposals"].update(proposals.as_dict())
            for gap in gap_suggestions:
                store.upsert_gap(gap.id, gap.kind, gap.title, gap.as_dict())
            return len(gap_suggestions)

        report.gaps = run_stage("gap analysis", _suggest_gaps) or 0

        def _write_audit() -> bool:
            findings = conflict_checks.audit_existing(existing, resolver)
            # Advisory only, and after the deterministic audit: the model can
            # hide or soften a finding, never add one.  report.ai keeps the
            # original count, so a model quietly dismissing real conflicts
            # shows up on the Status page rather than disappearing.
            if provider is not None and options.llm_audit and findings:
                verdicts = run_ai(
                    "audit_review", llm_audit.review, findings, existing, provider
                )
                if verdicts is not None:
                    before = len(findings)
                    findings = llm_audit.apply_verdicts(findings, verdicts)
                    report.ai["audit_review"].update(
                        {**verdicts.as_dict(), "findings_before": before,
                         "findings_after": len(findings)}
                    )
            store.set_meta("last_audit", str(int(time.time())))
            store.set_meta("existing_automation_audit", json.dumps(findings))
            return True

        run_stage("automation audit", _write_audit)

        # Last, and deliberately so: it can only announce suggestions that are
        # already persisted and readable.
        report.notified = run_stage(
            "notification", notifier.announce, store, client, options, run_id
        ) or {}
        if report.notified.get("error"):
            report.degradations.append(
                f"Could not tell Home Assistant about new suggestions: "
                f"{report.notified['error']}"
            )

        # A run where some miners failed still produced results, but saying it
        # was plain "ok" would hide that from the user.
        report.status = "partial" if (report.miner_errors or report.ai_errors) else "ok"
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
