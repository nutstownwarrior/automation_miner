"""Two rules that cannot both apply are not in conflict.

The audit compared triggers and targets and never looked at conditions, while
its message claimed the pair applied "under overlapping conditions" - so
complementary automations ("when I am out" / "when I am in") were reported as
conflicts on the strength of a claim nothing had checked.
"""

from __future__ import annotations

import pytest
from amminer.conditions import describe, provably_exclusive


def state(entity_id, value):
    return {"condition": "state", "entity_id": entity_id, "state": value}


def numeric(entity_id, **bounds):
    return {"condition": "numeric_state", "entity_id": entity_id, **bounds}


def when(**kwargs):
    return {"condition": "time", **kwargs}


@pytest.mark.parametrize(
    "label, first, second",
    [
        ("opposite states", [state("person.alex", "home")], [state("person.alex", "not_home")]),
        ("disjoint ranges", [numeric("sensor.lux", below=20)], [numeric("sensor.lux", above=500)]),
        ("ranges that touch", [numeric("sensor.lux", below=20)], [numeric("sensor.lux", above=20)]),
        ("weekday vs weekend", [when(weekday=["mon", "tue"])], [when(weekday=["sat", "sun"])]),
        ("morning vs evening",
         [when(after="06:00:00", before="09:00:00")],
         [when(after="18:00:00", before="22:00:00")]),
        ("overnight vs midday",
         [when(after="22:00:00", before="06:00:00")],
         [when(after="12:00:00", before="14:00:00")]),
        ("sun up vs sun down",
         [{"condition": "sun", "after": "sunrise"}],
         [{"condition": "sun", "after": "sunset"}]),
        ("one of several contradicts",
         [state("person.alex", "home"), numeric("sensor.lux", below=20)],
         [state("person.alex", "not_home")]),
    ],
)
def test_provably_exclusive_pairs(label, first, second):
    assert provably_exclusive(first, second) is True, label
    assert provably_exclusive(second, first) is True, f"{label} (reversed)"


@pytest.mark.parametrize(
    "label, first, second",
    [
        ("identical", [state("person.alex", "home")], [state("person.alex", "home")]),
        ("different entities", [state("person.alex", "home")], [state("person.sam", "not_home")]),
        ("overlapping state lists",
         [state("x.y", ["home", "work"])], [state("x.y", ["work"])]),
        ("overlapping ranges",
         [numeric("sensor.lux", below=20)], [numeric("sensor.lux", above=5)]),
        ("overlapping weekdays", [when(weekday=["mon", "tue"])], [when(weekday=["tue"])]),
        ("a window inside a wrapping one",
         [when(after="22:00:00", before="06:00:00")],
         [when(after="23:00:00", before="23:30:00")]),
        ("one side has none", [], [state("person.alex", "home")]),
        ("a template it cannot read",
         [{"condition": "template", "value_template": "{{ true }}"}],
         [state("person.alex", "home")]),
    ],
)
def test_not_proven_exclusive(label, first, second):
    """'Cannot prove exclusive' must never be reported as 'they overlap'."""
    assert provably_exclusive(first, second) is False, label
    assert provably_exclusive(second, first) is False, f"{label} (reversed)"


def test_mined_conditions_compare_against_raw_ones():
    """A candidate carries dataclass conditions; an existing rule carries dicts."""
    from amminer.miners.base import Condition

    mined = [Condition(kind="state", entity_id="person.alex", state="home")]
    assert provably_exclusive(mined, [state("person.alex", "not_home")]) is True
    assert provably_exclusive(mined, [state("person.alex", "home")]) is False


def test_describe_is_readable():
    assert "person.alex is home" in describe([state("person.alex", "home")])
    assert "above 8" in describe([numeric("sensor.t", above=8)])
    assert describe([]) == "no conditions"
