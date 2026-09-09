"""Options handling, and the add-on packaging metadata itself."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from amminer.config import Options
from amminer.scheduler import next_run_at
from amminer.version import __version__

ADDON_DIR = Path(__file__).resolve().parents[1] / "automation_miner"


# --- options ------------------------------------------------------------
def test_zero_config_defaults_are_usable():
    options = Options.load("/definitely/not/a/file.json")
    assert options.llm_provider == "none"
    assert options.min_consistency == 0.6
    assert options.override_window_seconds == 120
    assert options.backtest_min_precision == 0.7
    assert options.backtest_max_false_fires_per_week == 3
    assert options.analysis_window_days == 0  # auto
    assert options.schedule


def test_options_are_read_from_the_supervisor_file(tmp_path):
    path = tmp_path / "options.json"
    path.write_text(json.dumps({"min_consistency": 0.9, "llm_provider": "ollama",
                                "excluded_entities": ["light.decoration"]}))
    options = Options.load(path)
    assert options.min_consistency == 0.9
    assert options.llm_provider == "ollama"
    assert "light.decoration" in options.excluded_entities


def test_malformed_options_fall_back_to_defaults(tmp_path):
    path = tmp_path / "options.json"
    path.write_text("{not json")
    assert Options.load(path).min_consistency == 0.6


def test_unknown_options_are_ignored():
    options = Options.from_mapping({"who_knows": 1, "min_consistency": 0.75})
    assert options.min_consistency == 0.75


def test_values_are_coerced_and_clamped():
    options = Options.from_mapping(
        {"min_consistency": "1.5", "min_occurrences": "1", "run_on_start": "false"}
    )
    assert options.min_consistency == 1.0
    assert options.min_occurrences == 2
    assert options.run_on_start is False


def test_default_exclusions_are_always_applied():
    options = Options(excluded_entities=["light.mine"])
    assert options.is_excluded("sensor.time") is True
    assert options.is_excluded("sensor.uptime") is True
    assert options.is_excluded("update.core") is True
    assert options.is_excluded("light.mine") is True
    assert options.is_excluded("light.kitchen") is False


def test_exclusions_accept_globs():
    options = Options(excluded_entities=["sensor.*_battery"])
    assert options.is_excluded("sensor.door_battery") is True
    assert options.is_excluded("sensor.door_state") is False


def test_env_overrides_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("AMMINER_HA_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("AMMINER_STATE_DIR", str(tmp_path / "state"))
    options = Options.load("/nope.json")
    assert options.ha_config_dir == str(tmp_path)
    assert options.state_dir == str(tmp_path / "state")


# --- scheduler -----------------------------------------------------------
def test_cron_expression_is_honoured():
    import datetime as dt

    after = dt.datetime(2024, 6, 1, 12, 0).timestamp()
    nxt = dt.datetime.fromtimestamp(next_run_at("0 3 * * *", after))
    assert (nxt.hour, nxt.minute) == (3, 0)
    assert nxt.timestamp() > after


def test_bad_cron_falls_back_to_daily():
    after = 1_700_000_000.0
    assert next_run_at("not a cron", after) == pytest.approx(after + 86400.0)


# --- add-on packaging ----------------------------------------------------
@pytest.fixture(scope="module")
def addon_config() -> dict:
    return yaml.safe_load((ADDON_DIR / "config.yaml").read_text())


def test_config_yaml_declares_ingress_correctly(addon_config):
    assert addon_config["ingress"] is True
    assert addon_config["ingress_port"] == 8099
    assert addon_config["slug"] == "automation_miner"
    # S6-overlay v3 requires init: false.
    assert addon_config["init"] is False


def test_config_yaml_maps_the_right_directories(addon_config):
    mappings = addon_config["map"]
    assert "homeassistant_config:ro" in mappings, "HA config must be mounted read-only"
    assert "addon_config:rw" in mappings, "the add-on needs its own private /config"


def test_config_yaml_grants_the_apis_we_use(addon_config):
    assert addon_config["hassio_api"] is True
    assert addon_config["homeassistant_api"] is True


def test_config_yaml_is_multi_arch(addon_config):
    assert set(addon_config["arch"]) >= {"amd64", "aarch64"}


def test_version_matches_the_package(addon_config):
    assert addon_config["version"] == __version__


def test_every_option_has_a_schema_entry(addon_config):
    options = set(addon_config["options"])
    schema = set(addon_config["schema"])
    assert options <= schema, f"options without a schema: {options - schema}"


def test_option_defaults_match_the_python_defaults(addon_config):
    defaults = Options()
    for key, value in addon_config["options"].items():
        if not hasattr(defaults, key) or isinstance(value, list):
            continue
        assert getattr(defaults, key) == value, f"{key} differs from the Python default"


def test_api_key_option_is_a_password_field(addon_config):
    assert addon_config["schema"]["llm_api_key"] == "password?"


def test_build_yaml_covers_every_declared_arch(addon_config):
    build = yaml.safe_load((ADDON_DIR / "build.yaml").read_text())
    assert set(build["build_from"]) >= set(addon_config["arch"])


def test_dockerfile_defaults_build_from(addon_config):
    """BUILD_FROM is no longer supplied by default, so the ARG needs a default."""
    dockerfile = (ADDON_DIR / "Dockerfile").read_text()
    assert "ARG BUILD_FROM=ghcr.io/home-assistant/base:latest" in dockerfile
    assert "FROM ${BUILD_FROM}" in dockerfile


def test_s6_service_is_wired_up():
    service = ADDON_DIR / "rootfs/etc/s6-overlay/s6-rc.d/automation_miner"
    assert (service / "run").is_file()
    assert (service / "type").read_text().strip() == "longrun"
    contents = ADDON_DIR / "rootfs/etc/s6-overlay/s6-rc.d/user/contents.d/automation_miner"
    assert contents.is_file(), "the service must be registered in the user bundle"


def test_requirements_pin_every_core_dependency():
    text = (ADDON_DIR / "requirements.txt").read_text()
    for package in ("fastapi", "SQLAlchemy", "pandas", "numpy", "scikit-learn", "mlxtend",
                    "PyMySQL", "pg8000", "voluptuous"):
        assert package.lower() in text.lower(), f"{package} is not pinned"
    # Heavy optional extras must NOT be hard requirements.
    for banned in ("torch", "stumpy==", "numba"):
        assert f"\n{banned}" not in text


def test_repository_yaml_is_valid():
    repository = yaml.safe_load((ADDON_DIR.parent / "repository.yaml").read_text())
    assert repository["name"] and repository["url"]
