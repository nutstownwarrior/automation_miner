"""Standing preferences learned from the reasons a person gave for saying no.

This is the only optional feature that can make a suggestion disappear, so the
tests are weighted towards what happens when the model is wrong: a preference
learned from a single dismissal, a suppression that names no preference, an id
that does not exist, and a preference the user has switched off.
"""

from __future__ import annotations

from amminer.llm import preferences as prefs
from amminer.llm.provider import LLMError, NullProvider
from amminer.miners.base import Action, Candidate, Trigger
from amminer.store import STATUS_NEW, STATUS_SUPPRESSED, Store


class StubLLM(NullProvider):
    name, enabled = "stub", True

    def __init__(self, payload=None, raises=False):
        self.payload = payload or {}
        self.raises = raises
        self.prompts: list[str] = []

    def complete_json(self, system, user):
        self.prompts.append(user)
        if self.raises:
            raise LLMError("stub is down")
        return self.payload


def dismissal(suggestion_id, reason, title="Turn on the guest room lamp"):
    return {"suggestion_id": suggestion_id, "title": title, "reason": reason}


def candidate(title="Turn on the guest room lamp", entity="light.guest"):
    return Candidate(
        miner="time_of_day",
        title=title,
        triggers=[Trigger(kind="time", at="22:00:00")],
        actions=[Action(service="light.turn_on", entity_id=entity)],
    )


THREE = [
    dismissal("d1", "We never automate the guest room, guests set it themselves."),
    dismissal("d2", "Guest room again - leave that room alone."),
    dismissal("d3", "The bedroom lights must never come on before we are awake."),
]


# --- learning ------------------------------------------------------------
def test_a_preference_needs_two_dismissals_behind_it():
    """One dismissal generalised is that dismissal with a wider blast radius."""
    llm = StubLLM({"preferences": [
        {"rule": "Never automate the guest room.", "from": ["d1", "d2"]},
        {"rule": "Never suggest anything at all.", "from": ["d3"]},
    ]})
    learned, error = prefs.learn(THREE, llm)
    assert error is None
    assert [p.rule for p in learned] == ["Never automate the guest room."]


def test_citations_that_do_not_exist_do_not_count():
    llm = StubLLM({"preferences": [
        {"rule": "Never automate the guest room.", "from": ["d1", "invented"]},
    ]})
    learned, _ = prefs.learn(THREE, llm)
    assert learned == []


def test_too_few_dismissals_never_reaches_the_model():
    llm = StubLLM({"preferences": [{"rule": "Anything", "from": ["d1", "d2"]}]})
    learned, error = prefs.learn(THREE[: prefs.MIN_DISMISSALS - 1], llm)
    assert (learned, error, llm.prompts) == ([], None, [])


def test_a_provider_failure_is_reported_not_raised():
    learned, error = prefs.learn(THREE, StubLLM(raises=True))
    assert learned == [] and "stub is down" in error


def test_at_most_eight_preferences_are_kept():
    llm = StubLLM({"preferences": [
        {"rule": f"Rule number {i}.", "from": ["d1", "d2"]} for i in range(20)
    ]})
    learned, _ = prefs.learn(THREE, llm)
    assert len(learned) == prefs.MAX_PREFERENCES


def test_only_the_titles_and_reasons_are_sent():
    """Never raw history: the prompt carries what the user wrote, nothing else."""
    llm = StubLLM({"preferences": []})
    prefs.learn(THREE, llm)
    assert "guest room" in llm.prompts[0]
    assert "states" not in llm.prompts[0] and "last_changed" not in llm.prompts[0]


# --- applying ------------------------------------------------------------
def _preference(rule="Never automate the guest room."):
    return prefs.Preference(rule=rule, evidence=["d1", "d2"])


def test_a_match_hides_the_candidate_and_says_which_rule_did_it():
    one = candidate()
    preference = _preference()
    llm = StubLLM({"matches": [
        {"id": one.id, "preference": preference.id, "reason": "This is the guest room."}
    ]})
    result = prefs.apply_preferences([one], [preference], llm)
    assert result.suppressed[one.id]["preference"] == preference.id
    assert result.suppressed[one.id]["rule"] == preference.rule


def test_a_suppression_naming_no_known_preference_is_refused():
    """An unaccountable disappearance is the failure mode this guards against."""
    one = candidate()
    llm = StubLLM({"matches": [{"id": one.id, "preference": "pmadeup", "reason": "x"}]})
    result = prefs.apply_preferences([one], [_preference()], llm)
    assert result.suppressed == {} and result.unsupported == 1


def test_an_invented_candidate_id_hides_nothing():
    one = candidate()
    llm = StubLLM({"matches": [{"id": "nope", "preference": _preference().id}]})
    result = prefs.apply_preferences([one], [_preference()], llm)
    assert result.suppressed == {} and result.unknown_ids == ["nope"]


def test_with_no_preferences_the_model_is_never_asked():
    llm = StubLLM({"matches": [{"id": "x", "preference": "p"}]})
    result = prefs.apply_preferences([candidate()], [], llm)
    assert (result.suppressed, llm.prompts) == ({}, [])


def test_a_provider_failure_hides_nothing():
    result = prefs.apply_preferences([candidate()], [_preference()], StubLLM(raises=True))
    assert result.suppressed == {} and "stub is down" in result.error


def test_a_malformed_reply_hides_nothing():
    result = prefs.apply_preferences([candidate()], [_preference()], StubLLM({"matches": "no"}))
    assert result.suppressed == {}


# --- the store side ------------------------------------------------------
def test_switching_a_preference_off_survives_the_next_run(tmp_path):
    """The one thing here the user decided is the one thing not recomputed."""
    store = Store(tmp_path / "s.db")
    preference = _preference()
    store.save_preferences([preference.as_dict()])
    assert store.deactivate_preference(preference.id)

    store.save_preferences([preference.as_dict()])  # the next run relearns it
    assert [p["active"] for p in store.list_preferences(active_only=False)] == [0]
    assert store.list_preferences() == []


def test_a_preference_no_longer_learned_is_dropped(tmp_path):
    store = Store(tmp_path / "s.db")
    store.save_preferences([_preference().as_dict()])
    store.save_preferences([_preference("Something else entirely.").as_dict()])
    assert [p["rule"] for p in store.list_preferences()] == ["Something else entirely."]


def test_switching_off_brings_back_everything_it_hid(tmp_path):
    store = Store(tmp_path / "s.db")
    preference = _preference()
    store.save_preferences([preference.as_dict()])
    store.upsert_suggestion(
        "s1", "time_of_day", "Guest lamp", "summary", 0.9,
        {"extra": {"suppressed_by": {"preference": preference.id, "rule": preference.rule}}},
        run_id=1,
    )
    store.set_status("s1", STATUS_SUPPRESSED)

    store.deactivate_preference(preference.id)
    assert store.get_suggestion("s1")["status"] == STATUS_NEW


def test_a_suggestion_hidden_by_another_preference_stays_hidden(tmp_path):
    store = Store(tmp_path / "s.db")
    keep, drop = _preference("Rule A."), _preference("Rule B.")
    store.save_preferences([keep.as_dict(), drop.as_dict()])
    store.upsert_suggestion(
        "s1", "time_of_day", "t", "s", 0.5,
        {"extra": {"suppressed_by": {"preference": keep.id, "rule": keep.rule}}}, run_id=1,
    )
    store.set_status("s1", STATUS_SUPPRESSED)

    store.deactivate_preference(drop.id)
    assert store.get_suggestion("s1")["status"] == STATUS_SUPPRESSED


def test_hidden_suggestions_are_pruned_like_new_ones(tmp_path):
    """Hidden is not decided, so a stale hidden row is litter just the same."""
    store = Store(tmp_path / "s.db")
    store.upsert_suggestion("s1", "m", "t", "s", 0.5, {}, run_id=1)
    store.set_status("s1", STATUS_SUPPRESSED)
    store.prune_suggestions(2)
    assert store.get_suggestion("s1") is None
