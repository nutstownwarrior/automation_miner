"""The ingress UI: rendering, ingress isolation, and the action endpoints."""

from __future__ import annotations

import pytest
from amminer.config import Options
from amminer.discovery.ha_config import HAConfig
from amminer.runner import Runner, candidate_from_payload
from amminer.web.app import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def wired(ha_config_dir, store, fake_client):
    options = Options(ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir))
    runner = Runner(options, store, fake_client)
    runner.ha_config = HAConfig(ha_config_dir)
    runner.run_now()
    app = create_app(options, store, runner=runner, client=fake_client, ingress_only=False)
    return TestClient(app), store, runner, fake_client


def test_every_page_renders(wired):
    client, _store, _runner, _ha = wired
    for path in ("/", "/gaps", "/audit", "/dismissed", "/status"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "Automation Miner" in response.text


def test_suggestion_detail_shows_evidence_and_backtest(wired):
    client, store, _runner, _ha = wired
    suggestion = store.list_suggestions(status="new")[0]
    response = client.get(f"/suggestion/{suggestion['id']}")
    assert response.status_code == 200
    assert "Why this was suggested" in response.text
    assert "Backtest against your real history" in response.text


def test_unknown_suggestion_is_404(wired):
    client, _store, _runner, _ha = wired
    assert client.get("/suggestion/does-not-exist").status_code == 404


def test_ingress_path_header_rewrites_every_link(wired):
    client, _store, _runner, _ha = wired
    prefix = "/api/hassio_ingress/TOKEN123"
    response = client.get("/", headers={"X-Ingress-Path": prefix})
    assert f'href="{prefix}/static/app.css"' in response.text
    assert f'href="{prefix}/gaps"' in response.text
    assert f'{prefix}/static/app.js' in response.text


def test_non_ingress_requests_are_refused(ha_config_dir, store, fake_client):
    """Only 172.30.32.2 may talk to the ingress port."""
    options = Options(ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir))
    app = create_app(options, store, runner=None, client=fake_client, ingress_only=True)
    client = TestClient(app)
    response = client.get("/")  # TestClient presents as 127.0.0.1 (testclient)
    assert response.status_code == 403
    assert "ingress only" in response.text


def asgi_get(app, path: str, client_host: str) -> tuple[int, str]:
    """Call the ASGI app directly with a chosen client address.

    Starlette's ``TestClient`` only grew a ``client=`` argument in recent
    versions, so building the scope ourselves keeps this test working across
    the whole supported range while still exercising the real middleware stack.
    """
    import asyncio

    messages: list[dict] = []

    async def run_request():
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"testserver")],
            "client": (client_host, 5000),
            "server": ("testserver", 80),
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        await app(scope, receive, send)

    asyncio.run(run_request())
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return status, body.decode()


def test_ingress_source_is_allowed(ha_config_dir, store, fake_client):
    options = Options(ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir))
    app = create_app(options, store, runner=None, client=fake_client, ingress_only=True)
    status, body = asgi_get(app, "/api/health", "172.30.32.2")
    assert status == 200
    assert '"status":"ok"' in body.replace(" ", "")


def test_any_other_source_is_refused(ha_config_dir, store, fake_client):
    options = Options(ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir))
    app = create_app(options, store, runner=None, client=fake_client, ingress_only=True)
    for host in ("127.0.0.1", "192.168.1.50", "172.30.32.3"):
        status, _body = asgi_get(app, "/api/health", host)
        assert status == 403, f"{host} must not reach the ingress port"


def test_health_endpoint(wired):
    client, _store, _runner, _ha = wired
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["running"] is False
    assert payload["counts"]["suggestions"] > 0


def test_dismiss_endpoint_persists(wired):
    client, store, _runner, _ha = wired
    suggestion_id = store.list_suggestions(status="new")[0]["id"]
    response = client.post(f"/api/suggestions/{suggestion_id}/dismiss", json={"reason": "nope"})
    assert response.status_code == 200
    assert store.get_suggestion(suggestion_id)["status"] == "dismissed"
    assert store.is_dismissed(suggestion_id)


def test_dismiss_without_a_body_works(wired):
    client, store, _runner, _ha = wired
    suggestion_id = store.list_suggestions(status="new")[0]["id"]
    assert client.post(f"/api/suggestions/{suggestion_id}/dismiss").status_code == 200


def test_restore_endpoint(wired):
    client, store, _runner, _ha = wired
    suggestion_id = store.list_suggestions(status="new")[0]["id"]
    client.post(f"/api/suggestions/{suggestion_id}/dismiss")
    client.post(f"/api/suggestions/{suggestion_id}/restore")
    assert store.get_suggestion(suggestion_id)["status"] == "new"


def test_shadow_endpoint(wired):
    client, store, _runner, _ha = wired
    suggestion_id = store.list_suggestions(status="new")[0]["id"]
    assert client.post(f"/api/suggestions/{suggestion_id}/shadow").status_code == 200
    assert store.get_suggestion(suggestion_id)["status"] == "shadow"


def test_yaml_endpoint_returns_renderable_yaml(wired):
    import yaml

    client, store, _runner, _ha = wired
    actionable = [s for s in store.list_suggestions(status="new") if s["payload"].get("actions")]
    response = client.get(f"/api/suggestions/{actionable[0]['id']}/yaml")
    assert response.status_code == 200
    parsed = yaml.safe_load(response.text)
    assert isinstance(parsed, list) and parsed[0]["alias"]


def test_apply_writes_and_reloads(wired):
    client, store, _runner, ha = wired
    actionable = [s for s in store.list_suggestions(status="new") if s["payload"].get("actions")]
    suggestion_id = actionable[0]["id"]
    result = client.post(f"/api/suggestions/{suggestion_id}/apply").json()

    assert result["ok"] is True, result
    assert result["written"] is True
    assert result["reloaded"] is True
    assert ha.written  # the automation reached the fake Core API
    assert ha.reloaded
    assert store.get_suggestion(suggestion_id)["status"] == "accepted"
    written = next(iter(ha.written.values()))
    assert written["trigger"] and written["action"]
    assert "id" not in written  # the id travels in the URL, per HA's config API


def test_apply_is_refused_when_validation_fails(wired):
    client, store, runner, ha = wired
    ha.check_config_result = "invalid"
    actionable = [s for s in store.list_suggestions(status="new") if s["payload"].get("actions")]
    result = client.post(f"/api/suggestions/{actionable[0]['id']}/apply").json()
    assert result["ok"] is False
    assert result["errors"]
    assert not ha.written


def test_apply_refuses_audit_findings(wired):
    client, store, _runner, _ha = wired
    audit = [s for s in store.list_suggestions(status="new") if not s["payload"].get("actions")]
    if not audit:
        pytest.skip("no audit-only findings in this fixture")
    result = client.post(f"/api/suggestions/{audit[0]['id']}/apply").json()
    assert result["ok"] is False


def test_gap_dismissal(wired):
    client, store, _runner, _ha = wired
    gap_id = store.list_gaps("new")[0]["id"]
    assert client.post(f"/api/gaps/{gap_id}/dismiss").status_code == 200
    assert gap_id not in {g["id"] for g in store.list_gaps("new")}


def test_run_endpoint_starts_an_analysis(wired):
    client, _store, _runner, _ha = wired
    assert client.post("/api/run").json()["status"] == "started"


def test_candidate_round_trips_through_the_store(wired):
    client, store, _runner, _ha = wired
    actionable = [s for s in store.list_suggestions(status="new") if s["payload"].get("actions")]
    payload = actionable[0]["payload"]
    rebuilt = candidate_from_payload(payload)
    # The identity must survive serialisation, or dismissals would not stick.
    assert rebuilt.id == payload["id"]
    assert rebuilt.actions and rebuilt.triggers
