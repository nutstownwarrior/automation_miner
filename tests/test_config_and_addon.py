"""Options handling, and the add-on packaging metadata itself."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from amminer.config import Options
from amminer.scheduler import next_run_at
from amminer.version import __version__

ROOT = Path(__file__).resolve().parents[1]
ADDON_DIR = ROOT / "automation_miner"


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


#: Keys the Supervisor already defaults; the add-on linter rejects restating them.
SUPERVISOR_DEFAULTS = {
    "boot": "auto",
    "startup": "application",
    "hassio_role": "default",
    "panel_admin": True,
    "ingress_port": 8099,
}


def test_config_yaml_declares_ingress_correctly(addon_config):
    assert addon_config["ingress"] is True
    assert addon_config["slug"] == "automation_miner"
    # S6-overlay v3 requires init: false.
    assert addon_config["init"] is False


def test_no_option_merely_restates_a_supervisor_default(addon_config):
    """The add-on linter fails the build on redundant keys, so assert it here."""
    restated = [key for key in SUPERVISOR_DEFAULTS if key in addon_config]
    assert restated == [], f"remove keys that only restate defaults: {restated}"


def test_the_served_port_matches_the_ingress_default(addon_config):
    """ingress_port is omitted, so the code must bind the Supervisor's default."""
    from amminer.__main__ import DEFAULT_PORT

    expected = addon_config.get("ingress_port", SUPERVISOR_DEFAULTS["ingress_port"])
    assert DEFAULT_PORT == expected


def test_build_yaml_has_no_redundant_args():
    build = yaml.safe_load((ADDON_DIR / "build.yaml").read_text())
    assert "args" not in build, "an empty args block is rejected by the add-on linter"


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


def test_ai_features_are_exposed_in_the_addon_ui(addon_config):
    """They must be togglable from HA's Configuration tab, not just code."""
    for option in ("llm_entity_classification", "llm_hypotheses", "llm_triage"):
        assert addon_config["options"][option] is False, f"{option} must default off"
        assert addon_config["schema"][option] == "bool?"


def test_ai_tuning_knobs_are_range_checked(addon_config):
    assert addon_config["schema"]["llm_triage_penalty"] == "float(0.0,1.0)?"
    assert addon_config["schema"]["llm_hypotheses_per_candidate"] == "int(0,10)?"


def test_api_key_option_is_a_password_field(addon_config):
    assert addon_config["schema"]["llm_api_key"] == "password?"


def _pinned_python_version() -> str:
    """The Python version build.yaml pins, e.g. "3.13" from a base-python tag."""
    build = yaml.safe_load((ADDON_DIR / "build.yaml").read_text())
    tags = list(build["build_from"].values())
    versions = set()
    for tag in tags:
        match = re.search(r":(\d+\.\d+)-alpine", tag)
        assert match, f"cannot read a Python version from {tag!r}"
        versions.add(match.group(1))
    assert len(versions) == 1, f"architectures disagree on the Python version: {versions}"
    return versions.pop()


def test_base_images_pin_an_explicit_python_version():
    """The plain Alpine base tracks whatever Python Alpine ships today.

    That silently moved the interpreter to 3.14 once already and invalidated
    every pinned wheel, so the base must name its Python version.
    """
    build = yaml.safe_load((ADDON_DIR / "build.yaml").read_text())
    for arch, image in build["build_from"].items():
        assert "base-python" in image, f"{arch} must use a base-python image"
        assert re.search(r":\d+\.\d+-alpine\d+\.\d+", image), (
            f"{arch} image tag must pin both Python and Alpine: {image}"
        )
    assert _pinned_python_version()


def test_wheel_check_targets_the_python_the_image_actually_runs():
    """The CI wheel job must verify the interpreter build.yaml pins.

    Verifying a different Python is worse than not verifying at all: it reports
    success while the image build fails on a missing wheel.
    """
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yaml").read_text())
    checked = str(workflow["jobs"]["wheels"]["steps"][-1]["env"]["PYTHON_VERSION"])
    assert checked == _pinned_python_version(), (
        f"CI checks wheels for Python {checked} but the image runs "
        f"{_pinned_python_version()}"
    )


def test_dockerfile_does_not_add_a_second_interpreter():
    """base-python images already ship Python; apk add python3 would duplicate it."""
    dockerfile = (ADDON_DIR / "Dockerfile").read_text()
    assert "command -v python3" in dockerfile, (
        "Python must only be installed when the base image lacks it"
    )


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


def _pinned(text: str) -> set[str]:
    """Package names actually pinned in a requirements file (comments ignored)."""
    names = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        names.add(line.split("==")[0].split("[")[0].strip().lower())
    return names


def test_requirements_pin_every_core_dependency():
    pinned = _pinned((ADDON_DIR / "requirements.txt").read_text())
    pinned |= _pinned((ADDON_DIR / "requirements-nodeps.txt").read_text())
    for package in ("fastapi", "sqlalchemy", "pandas", "numpy", "scipy", "mlxtend",
                    "pymysql", "pg8000", "voluptuous", "pydantic"):
        assert package in pinned, f"{package} is not pinned"


def test_heavy_optional_extras_are_not_hard_requirements():
    pinned = _pinned((ADDON_DIR / "requirements.txt").read_text())
    pinned |= _pinned((ADDON_DIR / "requirements-nodeps.txt").read_text())
    for banned in ("torch", "stumpy", "numba", "river"):
        assert banned not in pinned, f"{banned} must stay optional and lazily imported"


def test_scikit_learn_is_deliberately_excluded():
    """scikit-learn has no musllinux wheels, so it must not reach the image.

    Nothing imports it; it was only ever an mlxtend dependency, which is why
    mlxtend is installed with --no-deps. If this ever needs to change, the base
    image has to move to the Debian variant at the same time.
    """
    pinned = _pinned((ADDON_DIR / "requirements.txt").read_text())
    pinned |= _pinned((ADDON_DIR / "requirements-nodeps.txt").read_text())
    assert "scikit-learn" not in pinned
    assert "matplotlib" not in pinned

    sources = list((ADDON_DIR / "amminer").rglob("*.py"))
    offenders = [
        path.name
        for path in sources
        if "import sklearn" in path.read_text() or "from sklearn" in path.read_text()
    ]
    assert offenders == [], f"these modules import scikit-learn: {offenders}"


def test_mlxtend_is_installed_without_its_dependency_closure():
    nodeps = (ADDON_DIR / "requirements-nodeps.txt").read_text()
    assert "mlxtend" in _pinned(nodeps)
    dockerfile = (ADDON_DIR / "Dockerfile").read_text()
    assert "--no-deps -r requirements-nodeps.txt" in dockerfile


def test_dockerfile_hardcodes_no_site_packages_path():
    """Base images disagree about where site-packages lives.

    Alpine's python3 uses /usr/lib/pythonX.Y, base-python builds into
    /usr/local. A hardcoded path failed the image build once already.
    """
    dockerfile = (ADDON_DIR / "Dockerfile").read_text()
    assert "/usr/lib/python3" not in dockerfile
    assert "/usr/local/lib/python3" not in dockerfile


def test_dockerfile_smoke_imports_the_package():
    """A broken image must fail the build, not the user's first start."""
    dockerfile = (ADDON_DIR / "Dockerfile").read_text()
    assert "import amminer" in dockerfile
    assert "mlxtend.frequent_patterns" in dockerfile, (
        "the build must prove mlxtend works without its dependency closure"
    )


def test_dockerfile_installs_wheels_only():
    """No compiler is installed, so a source build would fail the image build."""
    dockerfile = (ADDON_DIR / "Dockerfile").read_text()
    assert "--only-binary=:all:" in dockerfile
    assert "build-base" not in dockerfile


def test_repository_yaml_is_valid():
    repository = yaml.safe_load((ADDON_DIR.parent / "repository.yaml").read_text())
    assert repository["name"] and repository["url"]
