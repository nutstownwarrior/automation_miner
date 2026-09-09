"""Discover and open Home Assistant's recorder database.

Order of preference:

1. ``recorder: db_url:`` from ``configuration.yaml`` (``!secret`` resolved).
2. The default SQLite database at ``<config>/home-assistant_v2.db``.

SQLite is always opened **read-only** and WAL-aware so we never take a write
lock away from Core.  MySQL/MariaDB and PostgreSQL URLs are rewritten onto
pure-python drivers (PyMySQL / pg8000) which need no compiler on Alpine.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, make_url

_LOGGER = logging.getLogger(__name__)

DEFAULT_SQLITE_NAME = "home-assistant_v2.db"

#: Drivers we swap in so the image needs no C toolchain.
_DRIVER_MAP = {
    "mysql": "mysql+pymysql",
    "mysql+mysqldb": "mysql+pymysql",
    "mysql+mysqlconnector": "mysql+pymysql",
    "mariadb": "mysql+pymysql",
    "mariadb+mysqldb": "mysql+pymysql",
    "postgresql": "postgresql+pg8000",
    "postgres": "postgresql+pg8000",
    "postgresql+psycopg2": "postgresql+pg8000",
    "postgresql+psycopg": "postgresql+pg8000",
}


def normalise_db_url(db_url: str) -> str:
    """Rewrite a recorder ``db_url`` onto a driver bundled with the add-on."""
    try:
        url = make_url(db_url)
    except Exception:  # noqa: BLE001 - malformed URL, hand it back unchanged
        return db_url
    driver = url.drivername.lower()
    replacement = _DRIVER_MAP.get(driver)
    if replacement is None:
        return db_url
    url = url.set(drivername=replacement)
    if replacement.startswith("postgresql"):
        # pg8000 does not understand libpq-only query args.
        query = {k: v for k, v in url.query.items() if k not in ("charset", "sslmode")}
        url = url.set(query=query)
    return url.render_as_string(hide_password=False)


def sqlite_readonly_url(path: str | Path) -> str:
    """Build a read-only, WAL-tolerant SQLite URL for *path*.

    ``mode=ro`` (not ``immutable=1``) is used deliberately: an immutable open
    ignores the ``-wal`` sidecar and would silently hide the most recent hours
    of history on a live system.
    """
    resolved = Path(path).resolve()
    uri = resolved.as_posix()
    return f"sqlite+pysqlite:///file:{uri}?mode=ro&uri=true"


@dataclass
class RecorderInfo:
    """Everything we learned about the recorder."""

    db_url: str | None = None
    dialect: str = "unknown"
    source: str = "none"  # configuration.yaml | default-sqlite | env
    available: bool = False
    error: str | None = None
    purge_keep_days: int = 10
    #: Post-2023 normalised schema (``states_meta`` / ``*_ts`` columns).
    normalised_schema: bool = False
    has_binary_context: bool = False
    has_text_context: bool = False
    has_statistics: bool = False
    tables: list[str] = field(default_factory=list)
    oldest_state_ts: float | None = None
    newest_state_ts: float | None = None

    @property
    def history_days(self) -> float:
        if self.oldest_state_ts is None or self.newest_state_ts is None:
            return 0.0
        return max((self.newest_state_ts - self.oldest_state_ts) / 86400.0, 0.0)

    @property
    def is_sqlite(self) -> bool:
        return self.dialect == "sqlite"

    def as_dict(self) -> dict:
        data = {
            "dialect": self.dialect,
            "source": self.source,
            "available": self.available,
            "error": self.error,
            "purge_keep_days": self.purge_keep_days,
            "normalised_schema": self.normalised_schema,
            "has_binary_context": self.has_binary_context,
            "has_text_context": self.has_text_context,
            "has_statistics": self.has_statistics,
            "history_days": round(self.history_days, 2),
            "oldest_state_ts": self.oldest_state_ts,
            "newest_state_ts": self.newest_state_ts,
        }
        if self.db_url:
            data["db_url"] = redact_url(self.db_url)
        return data


def redact_url(db_url: str) -> str:
    """Strip credentials so a URL can be shown in the UI / logs."""
    try:
        parts = urlsplit(db_url)
    except ValueError:
        return "<unparseable>"
    if not parts.netloc or "@" not in parts.netloc:
        return db_url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, f"***@{host}", parts.path, "", ""))


def discover_db_url(ha_config, state_dir: str | Path | None = None) -> tuple[str | None, str]:
    """Return ``(db_url, source)`` for the recorder database."""
    env_url = os.environ.get("AMMINER_RECORDER_URL")
    if env_url:
        return env_url, "env"

    configured = ha_config.recorder_db_url() if ha_config is not None else None
    if configured:
        if configured.startswith("sqlite"):
            try:
                path = make_url(configured).database
            except Exception:  # noqa: BLE001
                path = None
            if path:
                return sqlite_readonly_url(path), "configuration.yaml"
            return configured, "configuration.yaml"
        return normalise_db_url(configured), "configuration.yaml"

    if ha_config is not None:
        default_path = Path(ha_config.config_dir) / DEFAULT_SQLITE_NAME
        if default_path.is_file():
            return sqlite_readonly_url(default_path), "default-sqlite"
        return None, "missing-sqlite"
    return None, "none"


def create_recorder_engine(db_url: str) -> Engine:
    """Create a conservatively configured read-only engine."""
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if db_url.startswith("sqlite"):
        # A single long-lived read connection keeps the SQLite page cache warm
        # and avoids re-opening the WAL on every query.
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        kwargs["pool_recycle"] = 3600
    return create_engine(db_url, **kwargs)


def probe(engine: Engine, info: RecorderInfo) -> RecorderInfo:
    """Fill in schema/vintage details by inspecting the live database."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    info.tables = sorted(tables)
    info.normalised_schema = "states_meta" in tables
    info.has_statistics = "statistics" in tables

    if "states" in tables:
        columns = {c["name"] for c in inspector.get_columns("states")}
        info.has_binary_context = "context_id_bin" in columns
        info.has_text_context = "context_id" in columns
        ts_column = "last_updated_ts" if "last_updated_ts" in columns else None
        with engine.connect() as conn:
            if ts_column:
                row = conn.execute(
                    text(f"SELECT MIN({ts_column}), MAX({ts_column}) FROM states")
                ).one_or_none()
            else:  # pre-2022 schema stores ISO strings
                row = conn.execute(
                    text(
                        "SELECT MIN(last_updated), MAX(last_updated) FROM states"
                    )
                ).one_or_none()
            if row and row[0] is not None:
                info.oldest_state_ts = _coerce_ts(row[0])
                info.newest_state_ts = _coerce_ts(row[1])
    info.available = "states" in tables
    return info


def _coerce_ts(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    import datetime as _dt

    if isinstance(value, _dt.datetime):
        return value.timestamp()
    try:
        return _dt.datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return None


@dataclass
class Recorder:
    """Handle on the recorder database plus what we know about it."""

    engine: Engine | None
    info: RecorderInfo

    @property
    def available(self) -> bool:
        return self.engine is not None and self.info.available

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()


def open_recorder(ha_config, options=None) -> Recorder:
    """Discover, open and probe the recorder database.

    Never raises: a failure is reported through ``RecorderInfo.error`` so the
    rest of the pipeline can degrade gracefully.
    """
    info = RecorderInfo()
    if ha_config is not None:
        info.purge_keep_days = ha_config.purge_keep_days()

    db_url, source = discover_db_url(ha_config)
    info.source = source
    info.db_url = db_url
    if not db_url:
        info.error = "No recorder database found (no db_url and no home-assistant_v2.db)"
        _LOGGER.warning("%s", info.error)
        return Recorder(None, info)

    try:
        info.dialect = make_url(db_url).get_backend_name()
    except Exception:  # noqa: BLE001
        info.dialect = "unknown"

    try:
        engine = create_recorder_engine(db_url)
        probe(engine, info)
    except Exception as err:  # noqa: BLE001 - any driver error degrades gracefully
        info.error = f"{type(err).__name__}: {err}"
        _LOGGER.warning("Could not open recorder at %s: %s", redact_url(db_url), err)
        return Recorder(None, info)

    _LOGGER.info(
        "Recorder ready: %s (%s, %.1f days of raw history, normalised=%s)",
        redact_url(db_url),
        info.dialect,
        info.history_days,
        info.normalised_schema,
    )
    return Recorder(engine, info)
