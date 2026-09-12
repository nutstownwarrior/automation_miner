"""The three optional AI features.

Each is tested on the same two axes: does it contribute what it should, and can
it do damage when the model is wrong? The second matters more - every one of
these is off by default precisely because a model's output is not trustworthy,
so the tests lean on hallucinated ids, invented roles, unusable proposals and
malformed responses.
"""

from __future__ import annotations

import datetime as dt

import pytest
from amminer.backtest import backtest_all
from amminer.config import Options
from amminer.enrich.detect import SignalSet, detect_signals
from amminer.enrich.signals import SignalSeries, SignalStore
from amminer.entities import EntityResolver
from amminer.llm import classify as classify_mod
from amminer.llm import hypothesis as hypothesis_mod
from amminer.llm import triage as triage_mod
from amminer.llm.provider import LLMError, NullProvider
from amminer.miners.base import Action, Candidate, Trigger
from amminer.recorderdb.models import Cause, StateChange
from amminer.util.timeutil import local_tz

TZ = local_tz()
START = dt.datetime(2024, 3, 1, tzinfo=TZ)
DAYS = 28
WINDOW = (START.timestamp(), (START + dt.timedelta(days=DAYS)).timestamp())


class StubLLM(NullProvider):
    """A provider that returns canned payloads and counts its calls."""

    name = "stub"
    enabled = True

    def __init__(self, payload=None, raises: bool = False):
        self.payload = payload if payload is not None else {}
        self.raises = raises
        self.calls = 0
        self.prompts: list[str] = []

    def complete_json(self, system, user):
        self.calls += 1
        self.prompts.append(user)
        if self.raises:
            raise LLMError("connection refused")
        return self.payload(user) if callable(self.payload) else self.payload


def resolver_with(*states) -> EntityResolver:
    resolver = EntityResolver()
    resolver.merge_states(list(states))
    return resolver


def state(entity_id, value="on", **attributes):
    return {"entity_id": entity_id, "state": value, "attributes": attributes}


# ======================================================================
# 1. Entity classification
# ======================================================================
@pytest.fixture
def multilingual_resolver() -> EntityResolver:
    """Entities the English-centric regex detector cannot classify."""
    return resolver_with(
        state("sensor.stroomprijs", "0.24", friendly_name="Stroomprijs",
              unit_of_measurement="EUR/kWh"),
        state("switch.vaatwasser", "off", friendly_name="Vaatwasser"),
        state("sensor.buitentemperatuur", "7.5", friendly_name="Buitentemperatuur",
              unit_of_measurement="°C"),
        state("light.keuken", "on", friendly_name="Keuken"),
    )


def test_classifier_finds_signals_the_regex_misses(multilingual_resolver):
    signals = detect_signals(multilingual_resolver)
    assert signals.energy_price == []  # the Dutch name defeats the pattern matcher

    provider = StubLLM({"assignments": [
        {"entity_id": "sensor.stroomprijs", "roles": ["energy_price"]},
        {"entity_id": "sensor.buitentemperatuur", "roles": ["outdoor_temperature"]},
    ]})
    result = classify_mod.classify(multilingual_resolver, provider)
    classify_mod.apply_to_signals(signals, result)

    assert signals.energy_price == ["sensor.stroomprijs"]
    assert signals.outdoor_temperature == ["sensor.buitentemperatuur"]
    assert result.added_count == 2


def test_classifier_is_additive_and_never_removes(multilingual_resolver):
    """A role the deterministic detector found must survive a wrong model."""
    signals = SignalSet(energy_price=["sensor.known_price"], person=["person.alex"])
    provider = StubLLM({"assignments": [
        {"entity_id": "sensor.stroomprijs", "roles": ["energy_price"]},
    ]})
    result = classify_mod.classify(multilingual_resolver, provider)
    classify_mod.apply_to_signals(signals, result)

    assert "sensor.known_price" in signals.energy_price
    assert "sensor.stroomprijs" in signals.energy_price
    assert signals.person == ["person.alex"]  # untouched


def test_classifier_rejects_hallucinated_entities(multilingual_resolver):
    provider = StubLLM({"assignments": [
        {"entity_id": "sensor.does_not_exist", "roles": ["energy_price"]},
        {"entity_id": "sensor.stroomprijs", "roles": ["energy_price"]},
    ]})
    result = classify_mod.classify(multilingual_resolver, provider)
    assert "sensor.does_not_exist" not in result.assignments
    assert "sensor.does_not_exist" in result.unknown_entities
    assert result.assignments == {"sensor.stroomprijs": ["energy_price"]}


def test_classifier_rejects_invented_roles(multilingual_resolver):
    provider = StubLLM({"assignments": [
        {"entity_id": "sensor.stroomprijs", "roles": ["teleportation", "energy_price"]},
    ]})
    result = classify_mod.classify(multilingual_resolver, provider)
    assert result.assignments == {"sensor.stroomprijs": ["energy_price"]}
    assert "teleportation" in result.unknown_roles


def test_classifier_ignores_malformed_responses(multilingual_resolver):
    for payload in ({}, {"assignments": "nope"}, {"assignments": [1, 2]},
                    {"assignments": [{"entity_id": 5, "roles": []}]}):
        result = classify_mod.classify(multilingual_resolver, StubLLM(payload))
        assert result.assignments == {}


def test_classifier_never_sends_state_or_history(multilingual_resolver):
    """The model is told what a thing is, not what it has been doing."""
    provider = StubLLM({"assignments": []})
    classify_mod.classify(multilingual_resolver, provider)
    prompt = "\n".join(provider.prompts)
    assert "sensor.stroomprijs" in prompt
    assert "0.24" not in prompt  # the current state must not be there
    assert "last_updated" not in prompt


def test_classification_is_cached_on_the_inventory(multilingual_resolver, store):
    provider = StubLLM({"assignments": [
        {"entity_id": "sensor.stroomprijs", "roles": ["energy_price"]},
    ]})
    first = classify_mod.classify(multilingual_resolver, provider, store=store)
    assert first.from_cache is False
    calls_after_first = provider.calls

    second = classify_mod.classify(multilingual_resolver, provider, store=store)
    assert second.from_cache is True
    assert provider.calls == calls_after_first  # no second round-trip
    assert second.assignments == first.assignments


def test_cache_is_invalidated_when_entities_change(multilingual_resolver, store):
    provider = StubLLM({"assignments": []})
    classify_mod.classify(multilingual_resolver, provider, store=store)
    calls = provider.calls

    multilingual_resolver.merge_states([state("sensor.new_thing", "1")])
    classify_mod.classify(multilingual_resolver, provider, store=store)
    assert provider.calls > calls


def test_classifier_batches_large_inventories():
    resolver = resolver_with(*[state(f"sensor.thing_{i}", "1") for i in range(25)])
    provider = StubLLM({"assignments": []})
    classify_mod.classify(resolver, provider, batch_size=10)
    assert provider.calls == 3


def test_classifier_noops_without_a_provider(multilingual_resolver):
    result = classify_mod.classify(multilingual_resolver, NullProvider())
    assert result.assignments == {}
    assert "no LLM provider" in result.error


def test_classifier_survives_a_provider_error(multilingual_resolver):
    result = classify_mod.classify(multilingual_resolver, StubLLM(raises=True))
    assert result.assignments == {}
    assert "connection refused" in result.error


# ======================================================================
# 2. Hypotheses - propose, then measure
# ======================================================================
@pytest.fixture
def cold_evening_case():
    """A heater used only on cold days: a plain daily rule misfires half the time."""
    store = SignalStore()
    temperature = SignalSeries("sensor.outdoor_temp", numeric=True)
    presence = SignalSeries("person.alex")
    changes: list[StateChange] = []
    for day in range(DAYS):
        cold = day % 2 == 0
        temperature.add((START + dt.timedelta(days=day, hours=12)).timestamp(),
                        4.0 if cold else 18.0)
        presence.add((START + dt.timedelta(days=day, hours=9)).timestamp(), "home")
        if cold:
            change = StateChange(
                "switch.heater", "on",
                (START + dt.timedelta(days=day, hours=18)).timestamp(), old_state="off",
            )
            change.cause = Cause.HUMAN
            changes.append(change)
    store.add(temperature)
    store.add(presence)

    candidate = Candidate(
        miner="time_of_day",
        title="Turn on the heater at 18:00",
        triggers=[Trigger(kind="time", at="18:00:00")],
        actions=[Action(service="switch.turn_on", entity_id="switch.heater")],
        score=0.6,
    )
    options = Options(llm_hypotheses=True)
    # This fixture is about the rescue, not about holdout validation - the
    # exact 0.5 baseline precision below is what the hypothesis is rescuing it
    # from, and only holds over the whole alternating cold/warm window.
    passed, rejected = backtest_all(
        [candidate], changes, store, options, WINDOW, validate_holdout=False
    )
    assert not passed and rejected, "the baseline rule must be rejected for this test"
    return changes, store, options, rejected


def test_a_verified_hypothesis_rescues_a_rejected_rule(cold_evening_case):
    changes, store, options, rejected = cold_evening_case
    assert rejected[0].backtest["precision"] == pytest.approx(0.5)

    provider = StubLLM({"hypotheses": [
        {"reason": "people heat when it is cold outside",
         "conditions": [{"kind": "numeric_state", "entity_id": "sensor.outdoor_temp",
                         "below": 10}]},
    ]})
    result = hypothesis_mod.propose_and_verify(
        rejected, changes, store, options, WINDOW, provider
    )
    assert len(result.accepted) == 1
    rescued = result.accepted[0]
    assert rescued.backtest["precision"] == 1.0
    assert rescued.backtest["passed"] is True
    assert rescued.miner.endswith("+hypothesis")
    # An AI-proposed rule is measured the same as any other: gated on a real
    # holdout when there is enough history for one, and never left with a
    # backtest that looks unvalidated because nobody asked it to validate.
    assert rescued.backtest["holdout_evaluated"] is True
    assert rescued.backtest["validation"] == "holdout"
    assert rescued.backtest["validation_note"]
    # Provenance must be visible: this came from a model, and it was measured.
    assert rescued.extra["hypothesis"]["reason"] == "people heat when it is cold outside"
    assert rescued.extra["hypothesis"]["origin_candidate"] == rejected[0].id


def test_signal_catalogue_uses_the_training_store_not_the_full_one(cold_evening_case):
    """A numeric threshold must never be chosen with the holdout's own values
    in view.  When a training-only store is given, the model sees only it -
    not the full one, which here holds a value that would betray a leak."""
    changes, store, options, rejected = cold_evening_case
    train_only = SignalStore()
    training_range = SignalSeries("sensor.outdoor_temp", numeric=True)
    training_range.add(START.timestamp(), -3.0)  # nowhere in the real fixture
    train_only.add(training_range)
    provider = StubLLM({"hypotheses": []})
    hypothesis_mod.propose_and_verify(
        rejected, changes, store, options, WINDOW, provider, train_store=train_only
    )
    prompt = provider.prompts[0]
    assert '"min": -3.0' in prompt and '"max": -3.0' in prompt
    assert '"min": 4.0' not in prompt  # the full store's real range


def test_signal_catalogue_falls_back_to_the_measuring_store_without_one(cold_evening_case):
    """Direct callers that pass no train_store (this module's other tests)
    keep working exactly as before - the fallback is for them, not for a real
    run, which always has a training slice to hand."""
    changes, store, options, rejected = cold_evening_case
    provider = StubLLM({"hypotheses": []})
    hypothesis_mod.propose_and_verify(rejected, changes, store, options, WINDOW, provider)
    prompt = provider.prompts[0]
    assert '"min": 4.0' in prompt and '"max": 18.0' in prompt


def test_a_useless_hypothesis_is_measured_and_discarded(cold_evening_case):
    """A plausible-sounding but wrong condition must not reach the user."""
    changes, store, options, rejected = cold_evening_case
    provider = StubLLM({"hypotheses": [
        {"reason": "heating only happens when someone is home",
         "conditions": [{"kind": "state", "entity_id": "person.alex", "state": "home"}]},
    ]})
    result = hypothesis_mod.propose_and_verify(
        rejected, changes, store, options, WINDOW, provider
    )
    assert result.proposed == 1
    assert result.accepted == []  # always true, so it removes no false fires
    assert result.attempts[0]["accepted"] is False


@pytest.mark.parametrize(
    ("label", "conditions"),
    [
        ("hallucinated entity",
         [{"kind": "state", "entity_id": "sensor.invented", "state": "on"}]),
        ("unsimulatable kind",
         [{"kind": "template", "value": "{{ now() }}"}]),
        ("numeric with no bound",
         [{"kind": "numeric_state", "entity_id": "sensor.outdoor_temp"}]),
        ("empty condition set", []),
        ("too many conditions",
         [{"kind": "numeric_state", "entity_id": "sensor.outdoor_temp", "below": 1},
          {"kind": "numeric_state", "entity_id": "sensor.outdoor_temp", "below": 2},
          {"kind": "numeric_state", "entity_id": "sensor.outdoor_temp", "below": 3},
          {"kind": "numeric_state", "entity_id": "sensor.outdoor_temp", "below": 4}]),
        ("garbage", ["not a dict"]),
    ],
)
def test_unusable_proposals_are_rejected_before_backtesting(
    cold_evening_case, label, conditions
):
    changes, store, options, rejected = cold_evening_case
    provider = StubLLM({"hypotheses": [{"reason": label, "conditions": conditions}]})
    result = hypothesis_mod.propose_and_verify(
        rejected, changes, store, options, WINDOW, provider
    )
    assert result.proposed == 0, f"{label} should never have been measured"
    assert result.accepted == []
    assert result.rejected_invalid >= 1


@pytest.mark.parametrize(
    "conditions",
    [
        [{"kind": "time", "after": "evening"}],
        [{"kind": "time", "before": "sunset"}],
        [{"kind": "time", "after": "25:00"}],
        [{"kind": "time", "after": "12:70"}],
        [{"kind": "time", "after": "noon", "weekday": ["mon"]}],
        [{"kind": "time", "after": ""}],
    ],
)
def test_a_time_bound_that_does_not_parse_is_rejected(conditions):
    """It would be an unevaluable no-op here and invalid YAML once applied."""
    assert hypothesis_mod.parse_conditions(conditions, {"sensor.outdoor_temp"}) is None


def test_valid_time_bounds_are_normalised():
    parsed = hypothesis_mod.parse_conditions(
        [{"kind": "time", "after": "6:5", "before": "23:00:30"}], {"sensor.x"}
    )
    assert parsed is not None
    assert parsed[0].after == "06:05:00"
    assert parsed[0].before == "23:00:30"


def test_hypotheses_may_not_reference_the_entity_being_acted_on(cold_evening_case):
    """Explaining 'turn the heater on' with 'the heater is on' is circular."""
    changes, store, options, rejected = cold_evening_case
    provider = StubLLM({"hypotheses": [
        {"reason": "circular", "conditions": [
            {"kind": "state", "entity_id": "switch.heater", "state": "off"}]},
    ]})
    result = hypothesis_mod.propose_and_verify(
        rejected, changes, store, options, WINDOW, provider
    )
    assert result.accepted == []
    assert result.rejected_invalid >= 1
    assert "switch.heater" not in "".join(provider.prompts).split('"available_signals"')[-1]


def test_rules_with_no_false_fires_are_not_worth_a_hypothesis(cold_evening_case):
    """Adding a condition only ever removes fires, so it cannot fix a missed action."""
    changes, store, options, _rejected = cold_evening_case
    never_fires = Candidate(
        miner="test",
        title="never fires",
        triggers=[Trigger(kind="time", at="03:00:00")],
        actions=[Action(service="switch.turn_on", entity_id="switch.heater")],
    )
    never_fires.backtest = {"simulated": True, "false_fires": 0, "reason": "x"}
    provider = StubLLM({"hypotheses": []})
    result = hypothesis_mod.propose_and_verify(
        [never_fires], changes, store, options, WINDOW, provider
    )
    assert result.considered == 0
    assert provider.calls == 0


def test_hypothesis_count_is_capped(cold_evening_case):
    changes, store, options, rejected = cold_evening_case
    options.llm_hypotheses_per_candidate = 2
    provider = StubLLM({"hypotheses": [
        {"reason": f"idea {i}",
         "conditions": [{"kind": "numeric_state", "entity_id": "sensor.outdoor_temp",
                         "below": 10 + i}]}
        for i in range(6)
    ]})
    result = hypothesis_mod.propose_and_verify(
        rejected, changes, store, options, WINDOW, provider
    )
    assert result.proposed == 2


def test_hypotheses_noop_without_a_provider(cold_evening_case):
    changes, store, options, rejected = cold_evening_case
    result = hypothesis_mod.propose_and_verify(
        rejected, changes, store, options, WINDOW, NullProvider()
    )
    assert result.accepted == []
    assert "no LLM provider" in result.error


def test_hypotheses_survive_a_provider_error(cold_evening_case):
    changes, store, options, rejected = cold_evening_case
    result = hypothesis_mod.propose_and_verify(
        rejected, changes, store, options, WINDOW, StubLLM(raises=True)
    )
    assert result.accepted == []
    assert "connection refused" in result.error


def test_signal_catalogue_gives_ranges_not_history(cold_evening_case):
    """Ranges stop the model proposing 'below 10' for a kilowatt sensor."""
    changes, store, options, rejected = cold_evening_case
    provider = StubLLM({"hypotheses": []})
    hypothesis_mod.propose_and_verify(
        rejected, changes, store, options, WINDOW, provider
    )
    prompt = provider.prompts[0]
    assert '"min": 4.0' in prompt and '"max": 18.0' in prompt
    assert '"observed_states"' in prompt  # categorical signals list their values


# ======================================================================
# 3. Triage - advisory only
# ======================================================================
def _pair() -> tuple[Candidate, Candidate]:
    sensible = Candidate(
        miner="association", title="hallway light on arrival",
        triggers=[Trigger(kind="state", entity_id="person.alex", to_state="home")],
        actions=[Action(service="light.turn_on", entity_id="light.hallway")], score=0.9,
    )
    absurd = Candidate(
        miner="association", title="bedroom light when thermostat stops",
        triggers=[Trigger(kind="state", entity_id="climate.living_room", to_state="off")],
        actions=[Action(service="light.turn_on", entity_id="light.bedroom")], score=0.8,
    )
    return sensible, absurd


def test_triage_demotes_the_implausible_and_leaves_the_rest():
    sensible, absurd = _pair()
    provider = StubLLM({"reviews": [
        {"id": sensible.id, "verdict": "plausible", "reason": "these belong together"},
        {"id": absurd.id, "verdict": "implausible", "reason": "an evening coincidence"},
    ]})
    result = triage_mod.triage([sensible, absurd], provider)
    triage_mod.apply_verdicts([sensible, absurd], result, penalty=0.5)

    assert sensible.score == 0.9, "a plausible verdict must not promote"
    assert absurd.score == 0.4
    assert absurd.extra["triage"]["reason"] == "an evening coincidence"
    assert result.demoted == [absurd.id]


def test_triage_never_removes_a_candidate():
    sensible, absurd = _pair()
    candidates = [sensible, absurd]
    provider = StubLLM({"reviews": [
        {"id": c.id, "verdict": "implausible", "reason": "no"} for c in candidates
    ]})
    result = triage_mod.triage(candidates, provider)
    returned = triage_mod.apply_verdicts(candidates, result, penalty=0.0)
    assert len(returned) == 2
    # Even at a zero multiplier the evidence and backtest survive untouched.
    assert all(c.evidence is not None for c in returned)


def test_triage_cannot_raise_a_score():
    sensible, _absurd = _pair()
    provider = StubLLM({"reviews": [
        {"id": sensible.id, "verdict": "plausible", "reason": "great idea"},
    ]})
    result = triage_mod.triage([sensible], provider)
    triage_mod.apply_verdicts([sensible], result, penalty=2.0)  # clamped to 1.0
    assert sensible.score == 0.9


def test_triage_ignores_unknown_ids_and_bogus_verdicts():
    sensible, absurd = _pair()
    provider = StubLLM({"reviews": [
        {"id": "not-a-real-id", "verdict": "implausible", "reason": "x"},
        {"id": sensible.id, "verdict": "banana", "reason": "x"},
        {"id": absurd.id, "verdict": "implausible", "reason": "fine"},
    ]})
    result = triage_mod.triage([sensible, absurd], provider)
    triage_mod.apply_verdicts([sensible, absurd], result, penalty=0.5)

    assert "not-a-real-id" in result.unknown_ids
    assert result.invalid_verdicts == 1
    assert "triage" not in sensible.extra
    assert sensible.score == 0.9
    assert absurd.score == 0.4


def test_triage_withholds_the_statistics_from_the_model():
    """It is asked whether the rule makes sense, not whether the numbers are good."""
    sensible, _absurd = _pair()
    sensible.evidence.consistency = 0.97
    sensible.backtest = {"precision": 0.99, "summary": "excellent"}
    provider = StubLLM({"reviews": []})
    triage_mod.triage([sensible], provider)
    prompt = provider.prompts[0]
    assert "0.97" not in prompt and "0.99" not in prompt
    assert "precision" not in prompt


# ======================================================================
# Bounding what a model writes into the database and the UI
# ======================================================================
PATHOLOGICAL = "why " * 5000


def test_model_text_helper_collapses_and_caps():
    from amminer.util.text import clean_model_text

    assert clean_model_text("  a\n\n  b  ") == "a b"
    assert clean_model_text(None, fallback="none") == "none"
    assert clean_model_text("") == ""
    capped = clean_model_text(PATHOLOGICAL, limit=50)
    assert len(capped) == 50 and capped.endswith("\u2026")


def test_a_pathological_hypothesis_reason_is_capped(cold_evening_case):
    """The reason is persisted and rendered, so it is bounded on entry."""
    changes, store, options, rejected = cold_evening_case
    provider = StubLLM({"hypotheses": [
        {"reason": PATHOLOGICAL,
         "conditions": [{"kind": "numeric_state", "entity_id": "sensor.outdoor_temp",
                         "below": 10}]},
    ]})
    result = hypothesis_mod.propose_and_verify(
        rejected, changes, store, options, WINDOW, provider
    )
    assert result.accepted
    stored = result.accepted[0].extra["hypothesis"]["reason"]
    assert len(stored) <= 300
    assert len(result.accepted[0].description) < 1000
    assert len(result.attempts[0]["reason"]) <= 300


def test_a_pathological_triage_reason_is_capped():
    sensible, _absurd = _pair()
    provider = StubLLM({"reviews": [
        {"id": sensible.id, "verdict": "implausible", "reason": PATHOLOGICAL},
    ]})
    result = triage_mod.triage([sensible], provider)
    triage_mod.apply_verdicts([sensible], result)
    assert len(sensible.extra["triage"]["reason"]) <= 300


def test_classification_summary_does_not_carry_the_whole_inventory():
    """as_dict() lands in runs.stats on every nightly run, so it stays bounded."""
    resolver = resolver_with(*[
        state(f"sensor.price_{i}", "1", friendly_name=f"Price {i}") for i in range(200)
    ])
    provider = StubLLM({"assignments": [
        {"entity_id": f"sensor.price_{i}", "roles": ["energy_price"]} for i in range(200)
    ]})
    result = classify_mod.classify(resolver, provider, batch_size=200)
    signals = SignalSet()
    classify_mod.apply_to_signals(signals, result)

    assert len(result.assignments) == 200  # the real mapping is still available
    summary = result.as_dict()
    assert "assignments" not in summary
    assert "added" not in summary
    assert summary["added_count"] == 200
    assert summary["assigned_entities"] == 200
    assert len(summary["added_sample"]) == 20

    import json

    assert len(json.dumps(summary)) < 2000, "a run record must not grow with the inventory"


def test_triage_noops_without_a_provider():
    sensible, _absurd = _pair()
    result = triage_mod.triage([sensible], NullProvider())
    assert result.verdicts == {}
    assert "no LLM provider" in result.error


def test_triage_survives_a_provider_error():
    sensible, _absurd = _pair()
    result = triage_mod.triage([sensible], StubLLM(raises=True))
    assert result.verdicts == {}
    assert "connection refused" in result.error
    assert sensible.score == 0.9
