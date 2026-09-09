"""The add-on's own SQLite state (never Home Assistant's database)."""

from .db import Store

__all__ = ["Store"]
