"""The Home Assistant Ingress UI.

Ingress proxies requests from the Supervisor at ``172.30.32.2`` and rewrites the
URL prefix per session, so:

* only that source address is accepted on the ingress port,
* every link is built from the ``X-Ingress-Path`` header rather than hard-coded.

The UI is server-rendered with Jinja2 and a little vanilla JS.  No CDN, no build
step - it has to work on an offline instance behind ingress.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import Options
from ..llm.provider import build_provider
from ..store import Store
from ..store.db import (
    STATUS_ACCEPTED,
    STATUS_DISMISSED,
    STATUS_NEW,
    STATUS_SHADOW,
    STATUS_SUPPRESSED,
)
from ..version import __version__

_LOGGER = logging.getLogger(__name__)

#: The Supervisor's ingress source address.  Nothing else may talk to us.
INGRESS_SOURCE = "172.30.32.2"

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


class IngressOnlyMiddleware(BaseHTTPMiddleware):
    """Reject anything that did not come through Home Assistant's ingress."""

    def __init__(self, app, enabled: bool = True, allowed: tuple[str, ...] = (INGRESS_SOURCE,)):
        super().__init__(app)
        self.enabled = enabled
        self.allowed = set(allowed)

    async def dispatch(self, request: Request, call_next):
        if self.enabled:
            client_host = request.client.host if request.client else None
            if client_host not in self.allowed:
                _LOGGER.warning("Rejected non-ingress request from %s", client_host)
                return PlainTextResponse("Forbidden: ingress only", status_code=403)
        return await call_next(request)


def ingress_path(request: Request) -> str:
    """The URL prefix Home Assistant is serving this session under."""
    return request.headers.get("X-Ingress-Path", "").rstrip("/")


def create_app(
    options: Options,
    store: Store,
    runner=None,
    client=None,
    ingress_only: bool = True,
) -> FastAPI:
    """Build the FastAPI application.

    ``runner`` is an object with ``run_now()``, ``last_report`` and ``resolver``
    - injected so the web layer never imports the pipeline directly and stays
    trivially testable.
    """
    app = FastAPI(title="Automation Miner", version=__version__, docs_url=None, redoc_url=None)
    app.add_middleware(IngressOnlyMiddleware, enabled=ingress_only)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.state.options = options
    app.state.store = store
    app.state.runner = runner
    app.state.client = client

    def context(request: Request, **extra: Any) -> dict[str, Any]:
        base = {
            "request": request,
            "base": ingress_path(request),
            "version": __version__,
            "options": options,
            "report": getattr(runner, "last_report", None),
            "counts": store.counts(),
        }
        base.update(extra)
        return base

    # --- pages --------------------------------------------------------
    # These are deliberately plain `def`, not `async def`.  Every one of them
    # talks to SQLite, and some talk to Home Assistant or to a language model
    # with a timeout measured in minutes.  An `async def` handler doing that
    # holds the single event loop for the whole call, so one detail page waiting
    # on a model freezes every other request - including /health, which is what
    # the Supervisor watches.  Starlette runs sync handlers in a threadpool.
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        suggestions = store.list_suggestions(status=[STATUS_NEW, STATUS_SHADOW])
        actionable = [s for s in suggestions if s["miner"] not in ("stale_automation", "unused_entity")]
        audit = [s for s in suggestions if s["miner"] in ("stale_automation", "unused_entity")]
        last_run = store.last_run()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=context(
                request,
                suggestions=actionable,
                audit=audit,
                last_run=last_run,
                gaps=store.list_gaps("new"),
            ),
        )

    @app.get("/suggestion/{suggestion_id}", response_class=HTMLResponse)
    def suggestion_detail(request: Request, suggestion_id: str):
        suggestion = store.get_suggestion(suggestion_id)
        if suggestion is None:
            raise HTTPException(status_code=404, detail="unknown suggestion")
        preview = None
        if runner is not None:
            preview = runner.preview_yaml(suggestion_id)
        return templates.TemplateResponse(
            request=request,
            name="suggestion.html",
            context=context(
                request,
                suggestion=suggestion,
                preview=preview,
                feedback=store.feedback_for(suggestion_id),
                shadow=store.shadow_report(suggestion_id),
            ),
        )

    @app.get("/audit", response_class=HTMLResponse)
    def audit_view(request: Request):
        raw = store.get_meta("existing_automation_audit", "[]")
        try:
            findings = json.loads(raw or "[]")
        except json.JSONDecodeError:
            findings = []
        return templates.TemplateResponse(
            request=request,
            name="audit.html",
            context=context(
                request,
                findings=findings,
                overrides=store.recent_overrides(50),
                override_counts=store.override_counts(),
            ),
        )

    @app.get("/gaps", response_class=HTMLResponse)
    def gaps_view(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="gaps.html",
            context=context(request, gaps=store.list_gaps()),
        )

    @app.get("/dismissed", response_class=HTMLResponse)
    def dismissed_view(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="dismissed.html",
            context=context(
                request,
                suggestions=store.list_suggestions(status=[STATUS_DISMISSED, STATUS_ACCEPTED]),
                # Nothing hides without a place to see it and switch it off.
                hidden=store.list_suggestions(status=STATUS_SUPPRESSED),
                preferences=store.list_preferences(active_only=False),
            ),
        )

    @app.get("/status", response_class=HTMLResponse)
    def status_view(request: Request):
        llm = build_provider(options).status().as_dict()
        return templates.TemplateResponse(
            request=request,
            name="status.html",
            context=context(
                request,
                runs=store.recent_runs(10),
                llm=llm,
                resolver_stats=getattr(runner, "resolver_stats", lambda: {})(),
            ),
        )

    # --- actions ------------------------------------------------------
    api = APIRouter(prefix="/api")

    async def _rule_from(request: Request) -> str:
        try:
            body = await request.json()
        except (ValueError, TypeError):
            return ""
        return str(body.get("rule") or "") if isinstance(body, dict) else ""

    @api.post("/run")
    async def trigger_run():
        if runner is None:
            raise HTTPException(status_code=503, detail="no analysis runner available")
        if runner.is_running:
            return JSONResponse({"status": "already_running"}, status_code=409)
        asyncio.get_running_loop().run_in_executor(None, runner.run_now)
        return {"status": "started"}

    @api.post("/suggestions/{suggestion_id}/dismiss")
    async def dismiss(request: Request, suggestion_id: str):
        suggestion = store.get_suggestion(suggestion_id)
        if suggestion is None:
            raise HTTPException(status_code=404, detail="unknown suggestion")
        # An optional {"reason": "..."} body; a bodyless POST is equally valid.
        reason = None
        try:
            body = await request.json()
            if isinstance(body, dict):
                reason = body.get("reason") or None
        except (ValueError, TypeError):
            pass
        store.dismiss(suggestion_id, reason, signature=suggestion_id)
        return {"status": "dismissed", "id": suggestion_id}

    @api.post("/suggestions/{suggestion_id}/restore")
    def restore(suggestion_id: str):
        if not store.restore(suggestion_id):
            raise HTTPException(status_code=404, detail="unknown suggestion")
        return {"status": "restored", "id": suggestion_id}

    # A learned preference is a guess about what someone meant, made from
    # sentences they typed in a hurry.  It is therefore theirs to correct, and
    # these five endpoints are what stop a bad generalisation from being
    # permanent.
    @api.post("/preferences")
    async def add_preference(request: Request):
        """Write a standing preference by hand, with no dismissals behind it."""
        preference = store.add_preference(await _rule_from(request))
        if preference is None:
            raise HTTPException(status_code=400, detail="a preference needs a rule")
        return {"status": "added", "preference": preference}

    @api.post("/preferences/{preference_id}")
    async def edit_preference(request: Request, preference_id: str):
        """Rewrite a preference.  The wording becomes the user's from here on."""
        preference = store.update_preference(preference_id, await _rule_from(request))
        if preference is None:
            if store.get_preference(preference_id) is None:
                raise HTTPException(status_code=404, detail="unknown preference")
            raise HTTPException(status_code=400, detail="a preference needs a rule")
        return {"status": "updated", "preference": preference}

    @api.post("/preferences/{preference_id}/delete")
    def delete_preference(preference_id: str):
        if not store.delete_preference(preference_id):
            raise HTTPException(status_code=404, detail="unknown preference")
        return {"status": "deleted", "id": preference_id}

    @api.post("/preferences/{preference_id}/on")
    def preference_on(preference_id: str):
        if not store.activate_preference(preference_id):
            raise HTTPException(status_code=404, detail="unknown preference")
        return {"status": "on", "id": preference_id}

    @api.post("/preferences/{preference_id}/off")
    def preference_off(preference_id: str):
        """Switch a preference off and bring back what it hid."""
        if not store.deactivate_preference(preference_id):
            raise HTTPException(status_code=404, detail="unknown preference")
        return {"status": "off", "id": preference_id}

    @api.post("/suggestions/{suggestion_id}/shadow")
    def shadow(suggestion_id: str):
        if store.get_suggestion(suggestion_id) is None:
            raise HTTPException(status_code=404, detail="unknown suggestion")
        store.set_status(suggestion_id, STATUS_SHADOW)
        store.add_feedback(suggestion_id, "shadow_started", {"ts": time.time()})
        return {"status": "shadow", "id": suggestion_id}

    @api.get("/suggestions/{suggestion_id}/yaml", response_class=PlainTextResponse)
    def suggestion_yaml(suggestion_id: str):
        if runner is None:
            raise HTTPException(status_code=503, detail="no runner available")
        preview = runner.preview_yaml(suggestion_id)
        if preview is None:
            raise HTTPException(status_code=404, detail="unknown suggestion")
        return PlainTextResponse(preview.get("yaml", ""), media_type="text/yaml")

    @api.post("/suggestions/{suggestion_id}/apply")
    def apply_suggestion(suggestion_id: str, confirm: bool = False):
        if runner is None:
            raise HTTPException(status_code=503, detail="no runner available")
        result = runner.apply(suggestion_id, confirm_conflicts=confirm)
        if result is None:
            raise HTTPException(status_code=404, detail="unknown suggestion")
        if result.get("ok"):
            store.set_status(suggestion_id, STATUS_ACCEPTED)
            store.add_feedback(suggestion_id, "accepted", result)
        elif result.get("needs_confirmation"):
            # Not a failure: the user has not answered yet.
            store.add_feedback(suggestion_id, "apply_needs_confirmation", result)
        else:
            store.add_feedback(suggestion_id, "apply_failed", result)
        return result

    @api.post("/gaps/{gap_id}/dismiss")
    def dismiss_gap(gap_id: str):
        store.set_gap_status(gap_id, "dismissed")
        return {"status": "dismissed", "id": gap_id}

    @api.get("/suggestions")
    def list_suggestions(status: str | None = None):
        return store.list_suggestions(status=status.split(",") if status else None)

    @api.get("/health")
    async def health():
        last = store.last_run()
        return {
            "status": "ok",
            "version": __version__,
            "running": bool(runner and runner.is_running),
            "last_run": last,
            "counts": store.counts(),
        }

    app.include_router(api)
    return app
