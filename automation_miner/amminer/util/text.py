"""Helpers for free text that came back from a model.

Anything a model writes is unbounded by construction: it is persisted into the
add-on's own database and rendered into the UI, so it has to be collapsed and
capped at the point it enters the system rather than wherever it happens to be
displayed.
"""

from __future__ import annotations

#: Long enough for a real one-sentence explanation, short enough that a
#: pathological reply cannot bloat a stored suggestion payload.
DEFAULT_LIMIT = 300


def clean_model_text(value: object, limit: int = DEFAULT_LIMIT, fallback: str = "") -> str:
    """Collapse whitespace and truncate *value* to *limit* characters."""
    text = " ".join(str(value if value is not None else "").split())
    if not text:
        return fallback
    limit = max(int(limit), 1)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
