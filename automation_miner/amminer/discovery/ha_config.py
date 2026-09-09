"""Parse Home Assistant's ``configuration.yaml`` (with ``!secret`` support).

Home Assistant's YAML dialect carries custom tags (``!secret``, ``!include``,
``!include_dir_merge_list`` ...).  A stock ``yaml.safe_load`` blows up on those,
so we install permissive constructors: ``!secret`` resolves against
``secrets.yaml`` and the ``!include*`` family is resolved where cheap and
otherwise reduced to a placeholder.  We only need a handful of keys
(``recorder:``, ``homeassistant: time_zone`` ...), so partial fidelity is fine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

_LOGGER = logging.getLogger(__name__)

_INCLUDE_DIR_TAGS = {
    "!include_dir_list": list,
    "!include_dir_merge_list": list,
    "!include_dir_named": dict,
    "!include_dir_merge_named": dict,
}


class _Placeholder(dict):
    """Marker for a tag we deliberately did not resolve."""

    def __init__(self, tag: str, value: Any) -> None:
        super().__init__({"__unresolved__": tag, "value": value})


def load_secrets(config_dir: Path) -> dict[str, Any]:
    """Read ``secrets.yaml`` next to ``configuration.yaml``."""
    path = Path(config_dir) / "secrets.yaml"
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as err:
        _LOGGER.warning("Could not parse secrets.yaml: %s", err)
        return {}
    return data if isinstance(data, dict) else {}


def build_loader(config_dir: Path, secrets: dict[str, Any], depth: int = 0):
    """Return a SafeLoader subclass that understands HA's custom tags."""
    config_dir = Path(config_dir)

    class HALoader(yaml.SafeLoader):
        pass

    def _secret(loader: yaml.Loader, node: yaml.Node) -> Any:
        key = str(loader.construct_scalar(node))  # type: ignore[arg-type]
        if key not in secrets:
            _LOGGER.debug("secrets.yaml has no key %r", key)
        return secrets.get(key)

    def _include(loader: yaml.Loader, node: yaml.Node) -> Any:
        rel = str(loader.construct_scalar(node))  # type: ignore[arg-type]
        if depth >= 4:
            return _Placeholder("!include", rel)
        target = config_dir / rel
        if not target.is_file():
            return _Placeholder("!include", rel)
        return load_yaml_file(target, config_dir, secrets, depth + 1)

    def _include_dir(loader: yaml.Loader, node: yaml.Node) -> Any:
        rel = str(loader.construct_scalar(node))  # type: ignore[arg-type]
        tag = node.tag
        factory = _INCLUDE_DIR_TAGS.get(tag, list)
        directory = config_dir / rel
        if depth >= 4 or not directory.is_dir():
            return factory()
        merged_list: list[Any] = []
        merged_dict: dict[str, Any] = {}
        for item in sorted(directory.glob("*.yaml")):
            loaded = load_yaml_file(item, config_dir, secrets, depth + 1)
            if factory is list:
                if isinstance(loaded, list):
                    merged_list.extend(loaded)
                elif loaded is not None:
                    merged_list.append(loaded)
            else:
                if tag == "!include_dir_named":
                    merged_dict[item.stem] = loaded
                elif isinstance(loaded, dict):
                    merged_dict.update(loaded)
        return merged_list if factory is list else merged_dict

    def _unknown(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> Any:
        if isinstance(node, yaml.ScalarNode):
            return _Placeholder(tag_suffix, loader.construct_scalar(node))
        if isinstance(node, yaml.SequenceNode):
            return _Placeholder(tag_suffix, loader.construct_sequence(node))
        return _Placeholder(tag_suffix, None)

    HALoader.add_constructor("!secret", _secret)
    HALoader.add_constructor("!include", _include)
    for tag in _INCLUDE_DIR_TAGS:
        HALoader.add_constructor(tag, _include_dir)
    HALoader.add_multi_constructor("!", _unknown)
    return HALoader


def load_yaml_file(
    path: Path,
    config_dir: Path | None = None,
    secrets: dict[str, Any] | None = None,
    depth: int = 0,
) -> Any:
    """Load a single HA YAML file, tolerating its custom tags."""
    path = Path(path)
    config_dir = Path(config_dir or path.parent)
    if secrets is None:
        secrets = load_secrets(config_dir)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        _LOGGER.warning("Could not read %s: %s", path, err)
        return None
    try:
        return yaml.load(text, Loader=build_loader(config_dir, secrets, depth))
    except yaml.YAMLError as err:
        _LOGGER.warning("Could not parse %s: %s", path, err)
        return None


class HAConfig:
    """Lazily parsed view of Home Assistant's configuration directory."""

    def __init__(self, config_dir: str | Path = "/homeassistant") -> None:
        self.config_dir = Path(config_dir)
        self._config: dict[str, Any] | None = None
        self._secrets: dict[str, Any] | None = None

    @property
    def available(self) -> bool:
        return (self.config_dir / "configuration.yaml").is_file()

    @property
    def secrets(self) -> dict[str, Any]:
        if self._secrets is None:
            self._secrets = load_secrets(self.config_dir)
        return self._secrets

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            data = load_yaml_file(
                self.config_dir / "configuration.yaml", self.config_dir, self.secrets
            )
            self._config = data if isinstance(data, dict) else {}
        return self._config

    # ------------------------------------------------------------------
    def recorder_options(self) -> dict[str, Any]:
        """Return the ``recorder:`` block (``{}`` when not configured)."""
        recorder = self.config.get("recorder")
        return recorder if isinstance(recorder, dict) else {}

    def recorder_db_url(self) -> str | None:
        url = self.recorder_options().get("db_url")
        if isinstance(url, str) and url.strip():
            return url.strip()
        return None

    def purge_keep_days(self, default: int = 10) -> int:
        value = self.recorder_options().get("purge_keep_days", default)
        try:
            return max(int(value), 1)
        except (TypeError, ValueError):
            return default

    def time_zone(self) -> str | None:
        ha = self.config.get("homeassistant")
        if isinstance(ha, dict):
            tz = ha.get("time_zone")
            if isinstance(tz, str):
                return tz
        return None

    def automation_files(self) -> list[Path]:
        """Best-effort list of files that may contain automations."""
        found: list[Path] = []
        for name in ("automations.yaml", "automation.yaml"):
            candidate = self.config_dir / name
            if candidate.is_file():
                found.append(candidate)
        for sub in ("automations", "automation"):
            directory = self.config_dir / sub
            if directory.is_dir():
                found.extend(sorted(directory.rglob("*.yaml")))
        return found
