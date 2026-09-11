"""The ingress UI: rendering, ingress isolation, and the action endpoints."""

from __future__ import annotations

import time

import pytest
from amminer.config import Options
from amminer.discovery.ha_config import HAConfig
from amminer.runner import Runner, candidate_from_payload
from amminer.web.app import create_app
from fastapi.testclient import TestClient
from markupsafe import escape


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


def test_a_partially_failed_ai_step_is_not_shown_as_a_success(wired):
    """A feature can return results AND an error, so `ran` alone is not success."""
    client, _store, runner, _ha = wired
    runner.last_report.ai = {
        "entity_classification": {
            "requested": True, "ran": True, "added_count": 2,
            "error": "connection reset after the first batch",
        },
        "triage": {"requested": True, "ran": True, "reviewed": 4, "error": None},
    }
    import re

    body = client.get("/status").text
    assert "connection reset after the first batch" in body
    section = body[body.index("AI assistance"):]
    items = {}
    for item in re.findall(r"<li>.*?</li>", section, re.S):
        flat = " ".join(item.split())
        match = re.search(r'<span class="tag (\w+)">([^<]+)</span>', flat)
        if match:
            items[match.group(2).strip()] = match.group(1)

    # The step that errored must not wear a success tag; the clean one may.
    assert items["entity classification"] == "warning"
    assert items["triage"] == "ok"


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


def test_apply_refuses_a_conflicting_rule_until_it_is_confirmed(wired):
    """The conflict check was run, counted, and then not consulted by apply."""
    client, store, _runner, ha = wired
    actionable = [s for s in store.list_suggestions(status="new") if s["payload"].get("actions")]
    suggestion_id = actionable[0]["id"]
    stored = store.get_suggestion(suggestion_id)
    payload = dict(stored["payload"])
    payload["conflicts"] = [
        {
            "kind": "value_inconsistency",
            "severity": "error",
            "message": "'Bedtime dim' drives light.kitchen to 'off' at the same time.",
        }
    ]
    store.upsert_suggestion(
        suggestion_id, stored["miner"], stored["title"], stored.get("summary") or "",
        stored.get("score") or 0.0, payload, stored.get("run_id"),
    )

    result = client.post(f"/api/suggestions/{suggestion_id}/apply").json()
    assert result["ok"] is False
    assert result["needs_confirmation"] is True
    assert not ha.written
    assert store.get_suggestion(suggestion_id)["status"] != "accepted"

    confirmed = client.post(f"/api/suggestions/{suggestion_id}/apply?confirm=true").json()
    assert confirmed["ok"] is True, confirmed
    assert ha.written


def test_restore_is_not_undone_by_the_next_run(wired):
    client, store, runner, _ha = wired
    suggestion_id = store.list_suggestions(status="new")[0]["id"]
    client.post(f"/api/suggestions/{suggestion_id}/dismiss")
    assert store.get_suggestion(suggestion_id)["status"] == "dismissed"

    assert client.post(f"/api/suggestions/{suggestion_id}/restore").status_code == 200
    assert store.is_dismissed(suggestion_id) is False

    runner.run_now()
    still_there = store.get_suggestion(suggestion_id)
    assert still_there is not None
    assert still_there["status"] == "new"


def test_restoring_an_unknown_suggestion_is_404(wired):
    client, _store, _runner, _ha = wired
    assert client.post("/api/suggestions/nope/restore").status_code == 404


# --- what the optional AI features put on screen ------------------------
def test_a_hidden_suggestion_is_shown_with_the_rule_that_hid_it(wired):
    """Hiding something without saying so is not something this add-on does."""
    from amminer.store import STATUS_SUPPRESSED

    client, store, _runner, _ha = wired
    store.save_preferences([
        {"id": "pguest", "rule": "Never automate the guest room.", "evidence": ["d1", "d2"]}
    ])
    suggestion = store.list_suggestions(status="new")[0]
    payload = dict(suggestion["payload"])
    payload["extra"] = {
        "suppressed_by": {"preference": "pguest", "rule": "Never automate the guest room.",
                          "reason": "This is the guest room."}
    }
    store.upsert_suggestion(
        suggestion["id"], suggestion["miner"], suggestion["title"], suggestion["summary"],
        suggestion["score"], payload,
    )
    store.set_status(suggestion["id"], STATUS_SUPPRESSED)

    title = str(escape(suggestion["title"]))
    page = client.get("/dismissed").text
    assert "Hidden by a preference" in page
    assert "Never automate the guest room." in page
    assert title in page
    # And it is not on the suggestions page it was hidden from.
    assert title not in client.get("/").text


def test_switching_a_preference_off_restores_what_it_hid(wired):
    from amminer.store import STATUS_NEW, STATUS_SUPPRESSED

    client, store, _runner, _ha = wired
    store.save_preferences([{"id": "pguest", "rule": "No guest room.", "evidence": ["d1"]}])
    suggestion = store.list_suggestions(status="new")[0]
    payload = dict(suggestion["payload"])
    payload["extra"] = {"suppressed_by": {"preference": "pguest", "rule": "No guest room."}}
    store.upsert_suggestion(
        suggestion["id"], suggestion["miner"], suggestion["title"], suggestion["summary"],
        suggestion["score"], payload,
    )
    store.set_status(suggestion["id"], STATUS_SUPPRESSED)

    assert client.post("/api/preferences/pguest/off").status_code == 200
    assert store.get_suggestion(suggestion["id"])["status"] == STATUS_NEW
    assert store.list_preferences() == []


def test_switching_off_an_unknown_preference_is_404(wired):
    client, _store, _runner, _ha = wired
    assert client.post("/api/preferences/nope/off").status_code == 404


def test_a_plain_language_explanation_is_rendered(wired):
    client, store, _runner, _ha = wired
    suggestion = store.list_suggestions(status="new")[0]
    payload = dict(suggestion["payload"])
    payload["extra"] = {"explanation": "You did this on 30 of the 34 weekday mornings."}
    store.upsert_suggestion(
        suggestion["id"], suggestion["miner"], suggestion["title"], suggestion["summary"],
        suggestion["score"], payload,
    )
    page = client.get("/").text
    assert "You did this on 30 of the 34 weekday mornings." in page
    # The figures it paraphrases are still there underneath.
    assert "Evidence:" in page


def test_a_scene_card_says_it_was_measured_as_one_rule(wired):
    client, store, _runner, _ha = wired
    suggestion = store.list_suggestions(status="new")[0]
    payload = dict(suggestion["payload"])
    payload["extra"] = {"scene": {"name": "Bedtime", "reason": "The house shuts down.",
                                  "members": ["a", "b"],
                                  "member_titles": ["Kitchen off", "Hall off"]}}
    store.upsert_suggestion(
        suggestion["id"], suggestion["miner"], suggestion["title"], suggestion["summary"],
        suggestion["score"], payload,
    )
    page = client.get("/").text
    assert "backtested as a single rule" in page
    assert "Kitchen off" in page


def test_the_preferences_panel_is_editable(wired):
    """A guess about what someone meant has to be correctable in the UI."""
    client, store, _runner, _ha = wired
    store.save_preferences([
        {"id": "pguest", "rule": "Never automate the guest room.", "evidence": ["d1", "d2"]}
    ])
    page = client.get("/dismissed").text
    assert 'value="Never automate the guest room."' in page
    assert 'data-action="preference-save"' in page
    assert 'data-action="preference-delete"' in page
    assert 'data-action="preference-add"' in page


def test_editing_a_preference_through_the_ui_sticks(wired):
    client, store, _runner, _ha = wired
    store.save_preferences([{"id": "pguest", "rule": "Wrong.", "evidence": ["d1", "d2"]}])

    response = client.post("/api/preferences/pguest", json={"rule": "Right, actually."})
    assert response.status_code == 200
    assert store.list_preferences()[0]["rule"] == "Right, actually."
    # And the next run cannot put the model's wording back.
    store.save_preferences([{"id": "pguest", "rule": "Wrong.", "evidence": ["d1", "d2"]}])
    assert store.list_preferences()[0]["rule"] == "Right, actually."


def test_editing_a_preference_to_nothing_is_refused(wired):
    client, store, _runner, _ha = wired
    store.save_preferences([{"id": "pguest", "rule": "A rule.", "evidence": ["d1"]}])
    assert client.post("/api/preferences/pguest", json={"rule": "  "}).status_code == 400
    assert store.list_preferences()[0]["rule"] == "A rule."


def test_editing_an_unknown_preference_is_404(wired):
    client, _store, _runner, _ha = wired
    assert client.post("/api/preferences/nope", json={"rule": "x"}).status_code == 404


def test_deleting_and_re_enabling_a_preference(wired):
    client, store, _runner, _ha = wired
    store.save_preferences([{"id": "pguest", "rule": "A rule.", "evidence": ["d1"]}])

    assert client.post("/api/preferences/pguest/off").status_code == 200
    assert store.list_preferences() == []
    assert client.post("/api/preferences/pguest/on").status_code == 200
    assert len(store.list_preferences()) == 1
    assert client.post("/api/preferences/pguest/delete").status_code == 200
    assert store.list_preferences(active_only=False) == []
    assert client.post("/api/preferences/pguest/delete").status_code == 404


def test_a_preference_can_be_written_by_hand(wired):
    client, store, _runner, _ha = wired
    response = client.post("/api/preferences", json={"rule": "Nothing in the bathroom."})
    assert response.status_code == 200
    assert response.json()["preference"]["source"] == "user"
    assert client.post("/api/preferences", json={"rule": ""}).status_code == 400


# --- wording help, which proposes and never writes -----------------------
def _drafting_app(ha_config_dir, store, fake_client, monkeypatch, reply=None,
                  raises=False, delay=0.0):
    """The archive page with the wording helper on and a stub model behind it."""
    import amminer.web.app as web_app
    from amminer.llm.provider import LLMError, NullProvider

    class StubLLM(NullProvider):
        name, enabled = "stub", True

        def complete_json(self, system, user):
            if delay:
                time.sleep(delay)
            if raises:
                raise LLMError("stub is down")
            return reply if reply is not None else {}

    monkeypatch.setattr(web_app, "build_provider", lambda _options: StubLLM())
    options = Options(
        ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir),
        llm_provider="ollama", llm_preferences=True,
    )
    runner = Runner(options, store, fake_client)
    runner.ha_config = HAConfig(ha_config_dir)
    return create_app(options, store, runner=runner, client=fake_client, ingress_only=False)


def _with_drafting(ha_config_dir, store, fake_client, monkeypatch, **kwargs):
    app = _drafting_app(ha_config_dir, store, fake_client, monkeypatch, **kwargs)
    return TestClient(app), app


def test_the_wording_helper_proposes_and_stores_nothing(
    ha_config_dir, store, fake_client, monkeypatch
):
    client, _app = _with_drafting(
        ha_config_dir, store, fake_client, monkeypatch,
        reply={"rule": "Do not suggest anything for the guest room lamp."},
    )
    store.save_preferences([{"id": "pg", "rule": "Never automate the guest room.",
                             "evidence": ["d1", "d2"]}])

    response = client.post("/api/preferences/draft", json={
        "preference": "pg", "instruction": "only the lamp, not the whole room",
    })
    assert response.status_code == 200
    assert response.json()["rule"] == "Do not suggest anything for the guest room lamp."
    assert response.json()["saved"] is False
    # The stored rule is untouched until a person presses Save.
    assert store.list_preferences()[0]["rule"] == "Never automate the guest room."

    client.post("/api/preferences/pg", json={"rule": response.json()["rule"]})
    assert store.list_preferences()[0]["rule"] == (
        "Do not suggest anything for the guest room lamp."
    )
    assert store.list_preferences()[0]["edited"] == 1


def test_the_wording_helper_has_its_own_path_not_the_edit_one(
    ha_config_dir, store, fake_client, monkeypatch
):
    """`/preferences/draft` must not be read as editing a preference called 'draft'."""
    client, _app = _with_drafting(
        ha_config_dir, store, fake_client, monkeypatch, reply={"rule": "A worded rule."},
    )
    assert client.post(
        "/api/preferences/draft", json={"instruction": "nothing in the bathroom"}
    ).status_code == 200
    assert store.list_preferences(active_only=False) == []


def test_the_wording_helper_needs_the_feature_switched_on(wired):
    client, store, _runner, _ha = wired
    response = client.post("/api/preferences/draft", json={"instruction": "narrow it"})
    assert response.status_code == 503
    assert "llm_preferences" in response.json()["detail"]


def test_a_draft_for_an_unknown_preference_is_404(
    ha_config_dir, store, fake_client, monkeypatch
):
    client, _app = _with_drafting(
        ha_config_dir, store, fake_client, monkeypatch, reply={"rule": "x"},
    )
    assert client.post(
        "/api/preferences/draft", json={"preference": "nope", "instruction": "narrow it"}
    ).status_code == 404


def test_a_model_that_is_down_does_not_change_a_preference(
    ha_config_dir, store, fake_client, monkeypatch
):
    client, _app = _with_drafting(ha_config_dir, store, fake_client, monkeypatch, raises=True)
    store.save_preferences([{"id": "pg", "rule": "A rule.", "evidence": ["d1"]}])
    response = client.post(
        "/api/preferences/draft", json={"preference": "pg", "instruction": "narrow it"}
    )
    assert response.status_code == 503
    assert store.list_preferences()[0]["rule"] == "A rule."


def test_something_that_is_not_a_preference_is_refused(
    ha_config_dir, store, fake_client, monkeypatch
):
    client, _app = _with_drafting(
        ha_config_dir, store, fake_client, monkeypatch, reply={"rule": ""}
    )
    response = client.post(
        "/api/preferences/draft", json={"instruction": "what is the weather"}
    )
    assert response.status_code == 422


def test_a_slow_model_does_not_freeze_the_rest_of_the_ui(
    ha_config_dir, store, fake_client, monkeypatch
):
    """The one handler here that is `async def` must not hold the event loop.

    A provider call has a timeout measured in minutes. Awaiting it on the loop
    would stall every other request behind it, /health included - which is what
    the Supervisor watches to decide the add-on is alive.
    """
    import asyncio

    app = _drafting_app(
        ha_config_dir, store, fake_client, monkeypatch,
        reply={"rule": "A worded rule."}, delay=1.0,
    )

    async def call(method: str, path: str, body: bytes = b""):
        sent: list[dict] = []
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": method, "scheme": "http",
            "path": path, "raw_path": path.encode(), "query_string": b"",
            "root_path": "", "client": ("172.30.32.2", 5000),
            "server": ("testserver", 80),
            "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
        }

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            sent.append(message)

        await app(scope, receive, send)
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        return status

    async def both():
        # Timed from before the slow request is scheduled, not from the health
        # call: a handler that blocks the loop runs to completion first, so the
        # health call itself would look fast while the whole UI had stalled.
        started = time.monotonic()
        slow = asyncio.create_task(
            call("POST", "/api/preferences/draft", b'{"instruction": "narrow it"}')
        )
        await asyncio.sleep(0.1)  # let the slow request reach the provider
        health = await call("GET", "/api/health")
        elapsed = time.monotonic() - started
        answered_while_waiting = not slow.done()
        return health, elapsed, answered_while_waiting, await slow

    health_status, elapsed, answered_while_waiting, draft_status = asyncio.run(both())
    assert (health_status, draft_status) == (200, 200)
    assert answered_while_waiting, "/health only answered once the model call had finished"
    assert elapsed < 0.5, f"/health waited {elapsed:.2f}s behind a model call"
