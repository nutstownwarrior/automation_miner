"""Thin Home Assistant Core / Supervisor API client.

Inside an add-on with ``homeassistant_api: true`` the Core REST API is reachable
at ``http://supervisor/core/api`` using ``SUPERVISOR_TOKEN`` as the bearer
token.  The WebSocket API (used for the registry lists) lives at
``ws://supervisor/core/websocket``.

Every method degrades to ``None`` / ``[]`` on failure - the add-on must keep
working when Core is restarting or the token is missing.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from typing import Any

import httpx

_LOGGER = logging.getLogger(__name__)

DEFAULT_CORE_BASE = "http://supervisor/core/api"
DEFAULT_SUPERVISOR_BASE = "http://supervisor"
DEFAULT_WS_URL = "ws://supervisor/core/websocket"


class HAClient:
    """Blocking HTTP client for the Core and Supervisor APIs."""

    def __init__(
        self,
        token: str | None = None,
        core_base: str | None = None,
        supervisor_base: str | None = None,
        ws_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.token = token if token is not None else os.environ.get("SUPERVISOR_TOKEN", "")
        self.core_base = (core_base or os.environ.get("AMMINER_CORE_URL") or DEFAULT_CORE_BASE).rstrip("/")
        self.supervisor_base = (supervisor_base or DEFAULT_SUPERVISOR_BASE).rstrip("/")
        self.ws_url = ws_url or os.environ.get("AMMINER_WS_URL") or DEFAULT_WS_URL
        self.timeout = timeout
        self._client: httpx.Client | None = None
        self.last_error: str | None = None

    # ------------------------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self.token)

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                trust_env=False,  # never route add-on traffic through a proxy
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> HAClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        if not self.configured:
            self.last_error = "SUPERVISOR_TOKEN is not set"
            return None
        try:
            response = self.client.request(method, url, **kwargs)
        except httpx.HTTPError as err:
            self.last_error = f"{type(err).__name__}: {err}"
            _LOGGER.debug("HTTP %s %s failed: %s", method, url, err)
            return None
        if response.status_code >= 400:
            self.last_error = f"HTTP {response.status_code} for {url}: {response.text[:200]}"
            _LOGGER.debug("%s", self.last_error)
            return None
        self.last_error = None
        if not response.content:
            return {}
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.text

    # --- Core API -----------------------------------------------------
    def ping(self) -> bool:
        result = self._request("GET", f"{self.core_base}/")
        return isinstance(result, dict) and "message" in result

    def get_states(self) -> list[dict[str, Any]]:
        """``GET /api/states`` - the authoritative *full* entity set."""
        result = self._request("GET", f"{self.core_base}/states")
        return result if isinstance(result, list) else []

    def get_state(self, entity_id: str) -> dict[str, Any] | None:
        result = self._request("GET", f"{self.core_base}/states/{entity_id}")
        return result if isinstance(result, dict) else None

    def get_config(self) -> dict[str, Any]:
        result = self._request("GET", f"{self.core_base}/config")
        return result if isinstance(result, dict) else {}

    def get_services(self) -> list[dict[str, Any]]:
        result = self._request("GET", f"{self.core_base}/services")
        return result if isinstance(result, list) else []

    def service_index(self) -> set[str]:
        """Flat ``{"light.turn_on", ...}`` set of every callable service."""
        index: set[str] = set()
        for entry in self.get_services():
            domain = entry.get("domain")
            services = entry.get("services") or {}
            if isinstance(domain, str) and isinstance(services, dict):
                index.update(f"{domain}.{name}" for name in services)
        return index

    def call_service(self, domain: str, service: str, data: dict[str, Any] | None = None) -> Any:
        return self._request(
            "POST", f"{self.core_base}/services/{domain}/{service}", json=data or {}
        )

    def check_config(self) -> dict[str, Any]:
        """``POST /api/config/core/check_config``.

        Returns ``{"result": "valid"|"invalid", "errors": ...}``; a transport
        failure is reported as ``{"result": "unavailable"}`` so callers can tell
        "did not run" apart from "failed".
        """
        result = self._request("POST", f"{self.core_base}/config/core/check_config")
        if isinstance(result, dict):
            return result
        return {"result": "unavailable", "errors": self.last_error}

    def upsert_automation(self, automation_id: str, config: dict[str, Any]) -> bool:
        """Write an automation through the Core config API."""
        result = self._request(
            "POST", f"{self.core_base}/config/automation/config/{automation_id}", json=config
        )
        return result is not None

    def reload_automations(self) -> bool:
        return self.call_service("automation", "reload") is not None

    def get_error_log(self) -> str:
        result = self._request("GET", f"{self.core_base}/error_log")
        return result if isinstance(result, str) else ""

    # --- Supervisor API ----------------------------------------------
    def supervisor_info(self) -> dict[str, Any]:
        result = self._request("GET", f"{self.supervisor_base}/info")
        if isinstance(result, dict):
            return result.get("data") or {}
        return {}

    def addons(self) -> list[dict[str, Any]]:
        """Installed add-ons - used to auto-detect Ollama, MariaDB, ..."""
        result = self._request("GET", f"{self.supervisor_base}/addons")
        if isinstance(result, dict):
            data = result.get("data") or {}
            addons = data.get("addons")
            if isinstance(addons, list):
                return addons
        return []

    # --- WebSocket ----------------------------------------------------
    def ws_registry_lists(self) -> dict[str, list[dict[str, Any]]] | None:
        """Fetch the registries over WebSocket (fallback for ``.storage``).

        Uses ``config/{entity,device,area,label}_registry/list``.  Requires an
        admin token; returns ``None`` when the socket is unavailable so the
        caller can fall back further to ``/api/states``.
        """
        if not self.configured:
            return None
        try:
            from websockets.sync.client import connect  # type: ignore[import-not-found]
        except ImportError:
            _LOGGER.debug("websockets not installed; skipping WebSocket registry fallback")
            return None

        commands = {
            "entities": "config/entity_registry/list",
            "devices": "config/device_registry/list",
            "areas": "config/area_registry/list",
            "labels": "config/label_registry/list",
            "floors": "config/floor_registry/list",
        }
        out: dict[str, list[dict[str, Any]]] = {}
        try:
            with connect(self.ws_url, open_timeout=self.timeout) as socket:
                json.loads(socket.recv())  # auth_required
                socket.send(json.dumps({"type": "auth", "access_token": self.token}))
                auth = json.loads(socket.recv())
                if auth.get("type") != "auth_ok":
                    _LOGGER.warning("WebSocket auth failed: %s", auth.get("message"))
                    return None
                msg_id = 0
                for key, command in commands.items():
                    msg_id += 1
                    socket.send(json.dumps({"id": msg_id, "type": command}))
                    while True:
                        message = json.loads(socket.recv())
                        if message.get("id") != msg_id:
                            continue
                        if message.get("success"):
                            result = message.get("result")
                            out[key] = result if isinstance(result, list) else []
                        else:
                            out[key] = []
                        break
        except Exception as err:  # noqa: BLE001 - any socket error means "fall back"
            _LOGGER.debug("WebSocket registry fetch failed: %s", err)
            return None
        return out


def probe_url(url: str, timeout: float = 2.0) -> bool:
    """Cheap reachability probe used for Ollama auto-detection."""
    try:
        response = httpx.get(url, timeout=timeout, trust_env=False)
    except httpx.HTTPError:
        return False
    return response.status_code < 500


def iter_addon_hostnames(addons: Iterable[dict[str, Any]], name_fragment: str) -> list[str]:
    """Hostnames of installed add-ons whose slug/name matches *name_fragment*."""
    fragment = name_fragment.lower()
    hosts: list[str] = []
    for addon in addons:
        slug = str(addon.get("slug", ""))
        name = str(addon.get("name", ""))
        if fragment in slug.lower() or fragment in name.lower():
            hostname = addon.get("hostname") or slug.replace("_", "-")
            hosts.append(str(hostname))
    return hosts
