"""Telling Home Assistant when a run found something.

Suggestions live in this add-on's own database, so without this nothing outside
the ingress page ever knows a run happened.  A nightly analysis nobody is told
about is a nightly analysis nobody reads.

Two rules shape everything here:

*Only what is new.*  A run re-surfaces every suggestion that still holds, so
announcing "what this run produced" would announce the same rules every night.
Only suggestions seen for the first time are worth interrupting someone for.

*Never at the cost of the run.*  Notifying is the last thing a run does and the
least important thing it does.  Every failure here is reported and swallowed;
none of it can lose a suggestion that was already persisted.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .config import Options
from .store import Store

_LOGGER = logging.getLogger(__name__)

#: How many titles a message lists before summarising the rest.  A notification
#: is a nudge, not the page.
MAX_TITLES = 5

#: Stable, so a new run replaces the previous notification rather than stacking
#: another one beside it.
NOTIFICATION_ID = "automation_miner_new_suggestions"


def ingress_path() -> str | None:
    """Best-effort deep link to this add-on's page.

    The Supervisor names the container after the add-on's full slug, which is
    the slug the ingress path uses with dashes instead of underscores.  When
    that is not available the message simply names the add-on, which is never
    wrong - a link that 404s would be worse than no link.
    """
    hostname = (os.environ.get("HOSTNAME") or "").strip()
    if not hostname or "-" not in hostname:
        return None
    return f"/hassio/addon/{hostname.replace('-', '_')}/ingress"


def build_message(suggestions: list[dict[str, Any]]) -> tuple[str, str]:
    """``(title, markdown body)`` for a batch of new suggestions."""
    count = len(suggestions)
    noun = "suggestion" if count == 1 else "suggestions"
    title = f"Automation Miner: {count} new {noun}"

    lines = [
        f"**{s.get('title') or 'Untitled suggestion'}**" for s in suggestions[:MAX_TITLES]
    ]
    remaining = count - len(lines)
    if remaining > 0:
        lines.append(f"…and {remaining} more.")

    link = ingress_path()
    where = (
        f"[Open Automation Miner]({link}) to review them."
        if link
        else "Open the Automation Miner add-on to review them."
    )
    lines.append("")
    lines.append(where)
    lines.append(
        "Nothing has been applied — each one is backtested against your own "
        "history and waits for you."
    )
    return title, "\n".join(lines)


def _call(client, domain: str, service: str, data: dict[str, Any]) -> str | None:
    """Call a service; return an error string rather than raising."""
    try:
        if client.call_service(domain, service, data) is None:
            return client.last_error or f"{domain}.{service} did not succeed"
    except Exception as err:  # noqa: BLE001 - a notification must never end a run
        return f"{type(err).__name__}: {err}"
    return None


def announce(
    store: Store, client, options: Options, run_id: int
) -> dict[str, Any]:
    """Announce this run's genuinely new suggestions.  Never raises."""
    report: dict[str, Any] = {"new": 0, "notified": False, "service_called": False}

    wants_notification = bool(options.notify_on_new_suggestions)
    wants_service = bool((options.notify_service or "").strip())
    if not wants_notification and not wants_service:
        return report

    suggestions = store.suggestions_first_seen_in(run_id)
    report["new"] = len(suggestions)
    if not suggestions:
        return report

    if client is None or not getattr(client, "configured", False):
        report["error"] = "no Home Assistant API access, so nothing could be sent"
        return report

    title, message = build_message(suggestions)
    errors: list[str] = []

    if wants_notification:
        error = _call(
            client,
            "persistent_notification",
            "create",
            {"title": title, "message": message, "notification_id": NOTIFICATION_ID},
        )
        if error:
            errors.append(f"persistent notification: {error}")
        else:
            report["notified"] = True

    if wants_service:
        service = (options.notify_service or "").strip()
        domain, _, name = service.partition(".")
        if not domain or not name:
            errors.append(
                f"notify_service {service!r} is not a 'domain.service' name, e.g. "
                "notify.mobile_app_your_phone"
            )
        else:
            error = _call(client, domain, name, {"title": title, "message": message})
            if error:
                errors.append(f"{service}: {error}")
            else:
                report["service_called"] = True

    if errors:
        report["error"] = "; ".join(errors)
        _LOGGER.warning("Could not announce new suggestions: %s", report["error"])
    return report
