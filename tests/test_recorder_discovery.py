"""Recorder discovery: db_url parsing, driver normalisation, schema probing."""

from __future__ import annotations

import pytest
from amminer.discovery.ha_config import HAConfig
from amminer.discovery.recorder import (
    discover_db_url,
    normalise_db_url,
    open_recorder,
    redact_url,
    sqlite_readonly_url,
)


@pytest.mark.parametrize(
    ("given", "expected_prefix"),
    [
        ("mysql://ha:pw@core-mariadb/homeassistant", "mysql+pymysql://"),
        ("mysql+mysqldb://ha:pw@core-mariadb/homeassistant", "mysql+pymysql://"),
        ("mariadb://ha:pw@core-mariadb/homeassistant", "mysql+pymysql://"),
        ("postgresql://ha:pw@db/homeassistant", "postgresql+pg8000://"),
        ("postgresql+psycopg2://ha:pw@db/homeassistant", "postgresql+pg8000://"),
    ],
)
def test_drivers_normalised_to_pure_python(given, expected_prefix):
    assert normalise_db_url(given).startswith(expected_prefix)


def test_postgres_drops_libpq_only_query_args():
    url = normalise_db_url("postgresql://ha:pw@db/homeassistant?charset=utf8mb4&sslmode=require")
    assert "charset" not in url and "sslmode" not in url


def test_mysql_keeps_charset():
    url = normalise_db_url("mysql://ha:pw@core-mariadb/homeassistant?charset=utf8mb4")
    assert "charset=utf8mb4" in url


def test_sqlite_opened_read_only_and_wal_aware():
    url = sqlite_readonly_url("/homeassistant/home-assistant_v2.db")
    assert "mode=ro" in url and "uri=true" in url
    # immutable=1 would hide the WAL, so it must NOT be used.
    assert "immutable" not in url


def test_redact_url_hides_credentials():
    assert "pw" not in redact_url("mysql://ha:pw@core-mariadb/homeassistant")
    assert "core-mariadb" in redact_url("mysql://ha:pw@core-mariadb/homeassistant")


def test_configured_db_url_wins(tmp_path):
    (tmp_path / "configuration.yaml").write_text(
        "recorder:\n  db_url: mysql://ha:pw@core-mariadb/homeassistant\n"
    )
    url, source = discover_db_url(HAConfig(tmp_path))
    assert source == "configuration.yaml"
    assert url.startswith("mysql+pymysql://")


def test_defaults_to_sqlite_when_no_db_url(ha_config_dir):
    url, source = discover_db_url(HAConfig(ha_config_dir))
    assert source == "default-sqlite"
    assert "home-assistant_v2.db" in url and "mode=ro" in url


def test_missing_database_degrades_without_raising(tmp_path):
    (tmp_path / "configuration.yaml").write_text("homeassistant: {}\n")
    recorder = open_recorder(HAConfig(tmp_path))
    assert recorder.available is False
    assert recorder.info.error


def test_probe_detects_normalised_schema(recorder_info):
    assert recorder_info.available
    assert recorder_info.normalised_schema is True
    assert recorder_info.has_binary_context is True
    assert recorder_info.has_statistics is True
    assert recorder_info.history_days > 40


def test_open_recorder_end_to_end(ha_config_dir):
    recorder = open_recorder(HAConfig(ha_config_dir))
    try:
        assert recorder.available
        assert recorder.info.dialect == "sqlite"
        assert recorder.info.purge_keep_days == 60
        assert "pw" not in str(recorder.info.as_dict())
    finally:
        recorder.close()
