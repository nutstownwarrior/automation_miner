"""The add-on's own SQLite state (never Home Assistant's database)."""

from .db import (
    STATUS_ACCEPTED,
    STATUS_DISMISSED,
    STATUS_NEW,
    STATUS_SHADOW,
    Store,
)

__all__ = [
    "STATUS_ACCEPTED",
    "STATUS_DISMISSED",
    "STATUS_NEW",
    "STATUS_SHADOW",
    "Store",
]
