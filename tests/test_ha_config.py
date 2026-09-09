"""Parsing Home Assistant's YAML dialect: !secret, !include, unknown tags."""

from __future__ import annotations

from amminer.discovery.ha_config import HAConfig, load_yaml_file


def test_resolves_secret_in_db_url(tmp_path):
    (tmp_path / "secrets.yaml").write_text(
        "db_password: sup3rs3cret\nrecorder_url: mysql://ha:pw@core-mariadb/homeassistant\n"
    )
    (tmp_path / "configuration.yaml").write_text(
        "recorder:\n  db_url: !secret recorder_url\n  purge_keep_days: 30\n"
    )
    config = HAConfig(tmp_path)
    assert config.recorder_db_url() == "mysql://ha:pw@core-mariadb/homeassistant"
    assert config.purge_keep_days() == 30


def test_missing_secret_does_not_raise(tmp_path):
    (tmp_path / "configuration.yaml").write_text("recorder:\n  db_url: !secret nope\n")
    config = HAConfig(tmp_path)
    assert config.recorder_db_url() is None
    assert config.purge_keep_days() == 10  # documented default


def test_include_and_unknown_tags_survive(tmp_path):
    (tmp_path / "sensors.yaml").write_text("- platform: template\n  sensors: {}\n")
    (tmp_path / "configuration.yaml").write_text(
        "homeassistant:\n"
        "  time_zone: Europe/Berlin\n"
        "sensor: !include sensors.yaml\n"
        "template: !include_dir_merge_list templates\n"
        "weird: !some_future_tag value\n"
        "recorder:\n  purge_keep_days: 45\n"
    )
    config = HAConfig(tmp_path)
    assert config.time_zone() == "Europe/Berlin"
    assert config.purge_keep_days() == 45
    assert isinstance(config.config["sensor"], list)


def test_include_dir_merge_list(tmp_path):
    (tmp_path / "packages").mkdir()
    (tmp_path / "packages" / "a.yaml").write_text("- alias: one\n")
    (tmp_path / "packages" / "b.yaml").write_text("- alias: two\n")
    (tmp_path / "configuration.yaml").write_text("automation: !include_dir_merge_list packages\n")
    config = HAConfig(tmp_path)
    aliases = {item["alias"] for item in config.config["automation"]}
    assert aliases == {"one", "two"}


def test_unreadable_config_is_not_fatal(tmp_path):
    config = HAConfig(tmp_path / "does-not-exist")
    assert config.available is False
    assert config.config == {}
    assert config.recorder_db_url() is None


def test_malformed_yaml_returns_none(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [unclosed\n")
    assert load_yaml_file(bad) is None


def test_automation_files_discovered(tmp_path):
    (tmp_path / "configuration.yaml").write_text("{}\n")
    (tmp_path / "automations.yaml").write_text("[]\n")
    (tmp_path / "automations").mkdir()
    (tmp_path / "automations" / "extra.yaml").write_text("[]\n")
    files = {p.name for p in HAConfig(tmp_path).automation_files()}
    assert files == {"automations.yaml", "extra.yaml"}
