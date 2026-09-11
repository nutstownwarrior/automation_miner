"""Plain-language explanations, and the number check that gates them.

The evidence line on a card is a statistician's sentence, and the fields behind
it do not even mean the same thing between miners. A model rewriting it is
useful. A model *inventing a figure* in the sentence that explains why to trust
something is the worst error this feature could make, so the check is not that
the sentence reads well - it is that every number in it came from the evidence.
"""

from __future__ import annotations

from amminer.llm import explain as explain_mod
from amminer.llm.provider import LLMError, NullProvider
from amminer.miners.base import Action, Candidate, Evidence, Trigger


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


def candidate(**evidence):
    base = {"occurrences": 30, "opportunities": 34, "consistency": 0.88, "window_days": 28.0}
    base.update(evidence)
    one = Candidate(
        miner="time_of_day",
        title="Kitchen light at 06:30",
        triggers=[Trigger(kind="time", at="06:30:00")],
        actions=[Action(service="light.turn_on", entity_id="light.kitchen")],
        evidence=Evidence(**base),
    )
    one.evidence.samples = [1700000000.0, 1700086400.0]
    return one


def _explain(text, one=None):
    one = one or candidate()
    llm = StubLLM({"explanations": [{"id": one.id, "text": text}]})
    return one, explain_mod.explain([one], llm)


# --- the number check ----------------------------------------------------
def test_a_sentence_built_from_the_evidence_is_kept():
    one, result = _explain("You did this on 30 of the 34 mornings in the window.")
    assert result.texts[one.id].startswith("You did this on 30")


def test_a_percentage_the_ratio_supports_is_kept():
    one, result = _explain("You did this on 88% of the mornings.")
    assert one.id in result.texts


def test_an_invented_number_drops_the_whole_sentence():
    """Not repaired - dropped. A wrong figure here is worse than no sentence."""
    one, result = _explain("You did this on 47 of the 34 mornings.")
    assert result.texts == {} and "47" in result.rejected_numbers


def test_a_rounded_number_is_still_an_invented_one():
    one, result = _explain("You did this about 9 times a week.")
    assert result.texts == {}


def test_the_time_in_the_rule_is_fair_game():
    one, result = _explain("You switched it on at about 06:30 on 30 of 34 mornings.")
    assert one.id in result.texts


def test_backtest_figures_count_as_evidence():
    one = candidate()
    one.backtest = {"true_fires": 22, "false_fires": 1, "precision": 0.96, "missed": 3}
    _, result = _explain("It would have been right 22 times and wrong once.", one)
    assert one.id in result.texts


def test_invented_numbers_finds_exactly_what_is_unsupported():
    one = candidate()
    assert explain_mod.invented_numbers("30 of 34, but 91 is new", one) == ["91"]


# --- everything else -----------------------------------------------------
def test_an_invented_id_explains_nothing():
    llm = StubLLM({"explanations": [{"id": "nope", "text": "A sentence."}]})
    result = explain_mod.explain([candidate()], llm)
    assert result.texts == {} and result.unknown_ids == ["nope"]


def test_raw_timestamps_never_leave_the_process():
    """The samples are real times a person was at home doing things."""
    llm = StubLLM({"explanations": []})
    explain_mod.explain([candidate()], llm)
    assert "1700000000" not in llm.prompts[0] and "samples" not in llm.prompts[0]


def test_a_provider_failure_explains_nothing():
    result = explain_mod.explain([candidate()], StubLLM(raises=True))
    assert result.texts == {} and "stub is down" in result.error


def test_a_malformed_reply_explains_nothing():
    assert explain_mod.explain([candidate()], StubLLM({"explanations": 7})).texts == {}


def test_applying_changes_nothing_but_the_sentence():
    one = candidate()
    before = (one.score, one.evidence.as_dict(), one.backtest, [a.as_dict() for a in one.actions])
    llm = StubLLM({"explanations": [{"id": one.id, "text": "30 of 34 mornings."}]})
    explain_mod.apply_explanations([one], explain_mod.explain([one], llm))
    after = (one.score, one.evidence.as_dict(), one.backtest, [a.as_dict() for a in one.actions])
    assert one.extra["explanation"] == "30 of 34 mornings."
    assert before == after


def test_a_dropped_sentence_leaves_the_card_as_it_was():
    one = candidate()
    llm = StubLLM({"explanations": [{"id": one.id, "text": "This happened 999 times."}]})
    explain_mod.apply_explanations([one], explain_mod.explain([one], llm))
    assert "explanation" not in one.extra


# --- ways a number can be wrong that set membership alone misses ---------
def test_a_permitted_number_asserted_about_the_wrong_field_is_caught():
    """28 is a real figure here - the window length, not the occurrence count."""
    one, result = _explain("This fired 28 of the 34 times it could have.")
    assert result.texts == {}


def test_the_real_pair_still_passes():
    one, result = _explain("This fired 30 of the 34 times it could have.")
    assert one.id in result.texts


def test_a_ratio_is_not_permitted_rounded_to_a_bare_integer():
    """f"{0.88:.0f}" is "1", which let the commonest invented figure through."""
    one, result = _explain("This happened only 1 time this week.")
    assert result.texts == {}


def test_a_number_written_as_a_word_is_not_a_number_that_was_checked():
    one, result = _explain("You did this on thirty-four of the thirty-eight mornings.")
    assert result.texts == {}


def test_a_negative_number_does_not_pass_on_the_strength_of_its_magnitude():
    one, result = _explain("It happened -30 times in the window.")
    assert result.texts == {}


def test_a_correct_figure_with_a_thousands_separator_is_kept():
    """Splitting on the comma rejected a sentence that was exactly right."""
    one = candidate(occurrences=1234, opportunities=1300, consistency=None)
    _, result = _explain("This happened 1,234 times.", one)
    assert one.id in result.texts


def test_no_evidence_field_leaks_a_raw_timestamp():
    """A blocklist that pops `samples` misses the timestamps inside `extra`."""
    one = candidate()
    one.evidence.window_start_ts = 1699999000.0
    one.evidence.window_end_ts = 1700000000.0
    one.evidence.extra["motifs"] = [{"start_ts": 1699999999.123, "length": 4}]
    one.backtest = {
        "true_fires": 5, "total_fires": 6, "precision": 0.83,
        "fire_samples": [1699999111.0], "false_fire_samples": [1699999222.0],
    }
    llm = StubLLM({"explanations": []})
    explain_mod.explain([one], llm)
    prompt = llm.prompts[0]
    for leaked in ("1699999999", "1699999000", "1700000000", "1699999111", "1699999222"):
        assert leaked not in prompt, f"{leaked} reached the model"
