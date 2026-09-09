"""The ingress web UI (FastAPI + server-rendered HTML, no external assets)."""

from .app import create_app

__all__ = ["create_app"]
