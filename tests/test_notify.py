"""Telling Home Assistant a run found something.

Without this the only way to learn a nightly run produced anything is to open
the add-on's page, so the interesting cases here are the ones that decide
whether someone keeps the feature switched on: is it only *new* findings, does
it stay quiet when there is nothing, and can it take the run down with it.
"""

from __future__ import annotations

from amminer.config import Options
from amminer.discovery.ha_config import HAConfig
from amminer.notify import NOTIFICATION_ID, announce, build_message, ingress_path
from amminer.runner import Runner


def _add(store, run_id: int, suggestion_id: str, title: str = "Turn on the lamp", score=0.9):
    store.upsert_suggestion(
        suggestion_id, "time_of_day", title, "because you do", score,
        {"actions": [{"entity_id": "light.kitchen"}]}, run_id,
    )


def _notifications(client) -> list[dict]:
    return [
        data for name, data in client.service_calls
        if name == "persistent_notification.create"
    ]


# --- only what is new ---------------------------------------------------
def test_a_run_announces_only_what_it_saw_for_the_first_time(store, fake_client):
    first = store.start_run()
    _add(store, first, "a", "Kitchen light at 06:30")
    _add(store, first, "b", "Hallway light when you arrive")
    report = announce(store, fake_client, Options(), first)
    assert report["new"] == 2
    assert report["notified"] is True
    assert len(_notifications(fake_client)) == 1

    # A second run re-surfaces both; neither is news any more.
    fake_client.service_calls.clear()
    second = store.start_run()
    _add(store, second, "a", "Kitchen light at 06:30")
    _add(store, second, "b", "Hallway light when you arrive")
    report = announce(store, fake_client, Options(), second)
    assert report["new"] == 0
    assert report["notified"] is False
    assert _notifications(fake_client) == []

    # A third run finds something genuinely new: only that one is announced.
    third = store.start_run()
    _add(store, third, "a", "Kitchen light at 06:30")
    _add(store, third, "c", "Close the blinds at sunset")
    report = announce(store, fake_client, Options(), third)
    assert report["new"] == 1
    assert "Close the blinds at sunset" in _notifications(fake_client)[0]["message"]
    assert "Kitchen light" not in _notifications(fake_client)[0]["message"]


def test_a_dismissed_suggestion_is_never_announced_again(store, fake_client):
    first = store.start_run()
    _add(store, first, "a")
    announce(store, fake_client, Options(), first)
    store.dismiss("a", "no thanks", signature="a")

    fake_client.service_calls.clear()
    second = store.start_run()
    _add(store, second, "a")
    report = announce(store, fake_client, Options(), second)
    assert report["new"] == 0
    assert _notifications(fake_client) == []


def test_nothing_is_sent_when_a_run_finds_nothing(store, fake_client):
    run_id = store.start_run()
    report = announce(store, fake_client, Options(), run_id)
    assert report == {"new": 0, "notified": False, "service_called": False}
    assert fake_client.service_calls == []


# --- the message --------------------------------------------------------
def test_the_message_lists_a_few_and_counts_the_rest():
    many = [{"title": f"Suggestion {i}"} for i in range(12)]
    title, message = build_message(many)
    assert title == "Automation Miner: 12 new suggestions"
    assert "Suggestion 0" in message and "Suggestion 4" in message
    assert "Suggestion 5" not in message
    assert "and 7 more" in message
    # It must not read as though something has been switched on.
    assert "Nothing has been applied" in message


def test_one_suggestion_reads_as_one():
    title, _ = build_message([{"title": "Only this"}])
    assert title == "Automation Miner: 1 new suggestion"


def test_a_run_replaces_its_previous_notification(store, fake_client):
    run_id = store.start_run()
    _add(store, run_id, "a")
    announce(store, fake_client, Options(), run_id)
    assert _notifications(fake_client)[0]["notification_id"] == NOTIFICATION_ID


def test_the_deep_link_is_omitted_rather_than_guessed(monkeypatch):
    monkeypatch.delenv("HOSTNAME", raising=False)
    assert ingress_path() is None
    _, message = build_message([{"title": "x"}])
    assert "Open the Automation Miner add-on" in message
    assert "](" not in message  # no fabricated link

    monkeypatch.setenv("HOSTNAME", "local-automation-miner")
    assert ingress_path() == "/hassio/addon/local_automation_miner/ingress"


# --- the configured notify service --------------------------------------
def test_a_configured_notify_service_is_called_too(store, fake_client):
    run_id = store.start_run()
    _add(store, run_id, "a")
    options = Options(notify_service="notify.mobile_app_pixel")
    report = announce(store, fake_client, options, run_id)

    assert report["service_called"] is True
    names = [name for name, _ in fake_client.service_calls]
    assert "notify.mobile_app_pixel" in names
    assert "persistent_notification.create" in names


def test_no_service_is_called_when_none_is_configured(store, fake_client):
    run_id = store.start_run()
    _add(store, run_id, "a")
    report = announce(store, fake_client, Options(), run_id)
    assert report["service_called"] is False
    assert [name for name, _ in fake_client.service_calls] == [
        "persistent_notification.create"
    ]


def test_a_malformed_service_name_is_reported_not_called(store, fake_client):
    run_id = store.start_run()
    _add(store, run_id, "a")
    options = Options(notify_service="mobile_app_pixel")  # no domain
    report = announce(store, fake_client, options, run_id)

    assert report["service_called"] is False
    assert "notify.mobile_app_your_phone" in report["error"]
    # The notification still went out; one bad option must not cost the other.
    assert report["notified"] is True


def test_both_can_be_turned_off(store, fake_client):
    run_id = store.start_run()
    _add(store, run_id, "a")
    options = Options(notify_on_new_suggestions=False, notify_service="")
    report = announce(store, fake_client, options, run_id)
    assert report == {"new": 0, "notified": False, "service_called": False}
    assert fake_client.service_calls == []


# --- it can never cost the run ------------------------------------------
def test_a_failing_service_is_reported_and_swallowed(store, fake_client):
    run_id = store.start_run()
    _add(store, run_id, "a")
    fake_client.service_fails = True
    report = announce(store, fake_client, Options(notify_service="notify.gone"), run_id)

    assert report["notified"] is False
    assert report["service_called"] is False
    assert "service unavailable" in report["error"]


def test_a_client_that_raises_is_reported_and_swallowed(store, fake_client):
    run_id = store.start_run()
    _add(store, run_id, "a")

    def explode(domain, service, data=None):
        raise RuntimeError("the socket went away")

    fake_client.call_service = explode
    report = announce(store, fake_client, Options(), run_id)
    assert report["notified"] is False
    assert "the socket went away" in report["error"]


def test_no_api_access_says_so_instead_of_failing(store):
    run_id = store.start_run()
    _add(store, run_id, "a")
    report = announce(store, None, Options(), run_id)
    assert report["notified"] is False
    assert "no Home Assistant API access" in report["error"]


# --- end to end through a real run --------------------------------------
def test_a_real_run_announces_its_new_suggestions(ha_config_dir, store, fake_client):
    options = Options(ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir))
    runner = Runner(options, store, fake_client)
    runner.ha_config = HAConfig(ha_config_dir)

    report = runner.run_now()
    assert report.notified["new"] > 0
    assert report.notified["notified"] is True
    assert _notifications(fake_client)

    # The second run finds the same rules, so it says nothing.
    fake_client.service_calls.clear()
    second = runner.run_now()
    assert second.notified["new"] == 0
    assert _notifications(fake_client) == []


def test_a_broken_notification_does_not_break_the_run(ha_config_dir, store, fake_client):
    options = Options(ha_config_dir=str(ha_config_dir), state_dir=str(ha_config_dir))
    runner = Runner(options, store, fake_client)
    runner.ha_config = HAConfig(ha_config_dir)
    fake_client.service_fails = True

    report = runner.run_now()
    assert report.error is None
    assert report.surfaced > 0
    assert store.list_suggestions(status="new"), "suggestions must still be persisted"
    assert any("Could not tell Home Assistant" in d for d in report.degradations)
