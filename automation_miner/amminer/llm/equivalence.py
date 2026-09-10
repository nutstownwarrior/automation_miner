"""Does the model's automation still say what the mined rule said?

The reference gate in :mod:`amminer.llm.validate` asks whether everything the
automation names *exists*.  That is a real check and it is not the important
one: an automation made entirely of real entities and real services can still
be a completely different automation from the one whose evidence the user read.

The prompt tells the model it may improve the alias and description and nothing
else.  This module enforces that instruction instead of trusting it: the model's
config is reduced to a canonical form covering triggers, conditions and actions,
and compared against the same reduction of the deterministic rendering.  Any
divergence is a rejection, and the rejection names the field that diverged.

Without this, a model that is malicious, prompt-injected, or simply confused can
return ``lock.unlock`` on the front door under an alias about a kitchen light,
and every other check passes.
"""

from __future__ import annotations

from typing import Any

#: Fields the prompt explicitly allows the model to rewrite.
REWRITABLE = ("alias", "description")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _canonical(value: Any) -> Any:
    """Order-insensitive, shorthand-insensitive form of a config fragment."""
    if isinstance(value, dict):
        items = {}
        for key, item in value.items():
            key = str(key)
            # HA accepts several spellings for the same thing; fold them so a
            # cosmetic rewrite is not mistaken for a semantic change.
            if key == "action":
                key = "service"
            elif key == "trigger":
                key = "platform"
            elif key == "entity_id":
                key = "__entities__"
                item = sorted(str(v) for v in _as_list(item))
            elif key == "target" and isinstance(item, dict):
                # ``target: {entity_id: x}`` and the ``entity_id: x`` shorthand
                # address the same thing; flatten so one is not read as a
                # change of target.  area_id/device_id/label_id ride along.
                items.update(_canonical(item))
                continue
            if key in ("data", "target") and not item:
                continue
            items[key] = _canonical(item)
        return tuple(sorted(items.items(), key=lambda kv: kv[0]))
    if isinstance(value, list):
        return tuple(sorted((_canonical(v) for v in value), key=repr))
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # 21 and 21.0 are the same setpoint.
        return float(value)
    return str(value)


def semantic_form(config: dict[str, Any]) -> dict[str, Any]:
    """The parts of an automation that decide what it actually does."""
    return {
        "trigger": _canonical(_as_list(config.get("trigger") or config.get("triggers"))),
        "condition": _canonical(_as_list(config.get("condition") or config.get("conditions"))),
        "action": _canonical(_as_list(config.get("action") or config.get("actions"))),
        "mode": _canonical(config.get("mode") or "single"),
    }


def differences(produced: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    """Field names where *produced* means something different from *reference*."""
    left, right = semantic_form(produced), semantic_form(reference)
    return [field for field in left if left[field] != right[field]]


def matches(produced: dict[str, Any], reference: dict[str, Any]) -> bool:
    return not differences(produced, reference)


def describe_divergence(
    produced: dict[str, Any], reference: dict[str, Any]
) -> list[str]:
    """Human-readable errors for each field the model changed."""
    errors: list[str] = []
    for field in differences(produced, reference):
        got = _as_list(produced.get(field) or produced.get(field + "s"))
        expected = _as_list(reference.get(field) or reference.get(field + "s"))
        errors.append(
            f"the model changed the automation's {field}: the mined rule has "
            f"{expected!r} but the model returned {got!r}. Only "
            f"{' and '.join(REWRITABLE)} may be rewritten."
        )
    return errors
