"""Copy a synthetic SQLite recorder fixture into MariaDB/MySQL or PostgreSQL.

This exists so the *same* production SQL in :mod:`amminer.recorderdb.queries`
can be exercised against a second dialect.  Column types mirror Home Assistant's
own recorder models closely enough for the queries to behave identically
(binary context columns, float timestamps, TEXT attribute blobs).
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

_LOGGER = logging.getLogger(__name__)

#: (table, DDL) in dependency order.
MYSQL_DDL: tuple[tuple[str, str], ...] = (
    (
        "states_meta",
        """CREATE TABLE states_meta (
            metadata_id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
            entity_id VARCHAR(255),
            UNIQUE KEY ix_states_meta_entity_id (entity_id)
        )""",
    ),
    (
        "state_attributes",
        """CREATE TABLE state_attributes (
            attributes_id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
            hash BIGINT,
            shared_attrs LONGTEXT
        )""",
    ),
    (
        "states",
        """CREATE TABLE states (
            state_id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
            metadata_id INTEGER,
            state VARCHAR(255),
            attributes_id INTEGER,
            old_state_id INTEGER,
            last_updated_ts DOUBLE,
            last_changed_ts DOUBLE,
            last_reported_ts DOUBLE,
            context_id_bin VARBINARY(16),
            context_user_id_bin VARBINARY(16),
            context_parent_id_bin VARBINARY(16),
            origin_idx SMALLINT,
            KEY ix_states_metadata_id_last_updated_ts (metadata_id, last_updated_ts),
            KEY ix_states_last_updated_ts (last_updated_ts)
        )""",
    ),
    (
        "event_types",
        """CREATE TABLE event_types (
            event_type_id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
            event_type VARCHAR(64),
            UNIQUE KEY ix_event_types_event_type (event_type)
        )""",
    ),
    (
        "event_data",
        """CREATE TABLE event_data (
            data_id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
            hash BIGINT,
            shared_data LONGTEXT
        )""",
    ),
    (
        "events",
        """CREATE TABLE events (
            event_id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
            event_type_id INTEGER,
            data_id INTEGER,
            origin_idx SMALLINT,
            time_fired_ts DOUBLE,
            context_id_bin VARBINARY(16),
            context_user_id_bin VARBINARY(16),
            context_parent_id_bin VARBINARY(16),
            KEY ix_events_time_fired_ts (time_fired_ts)
        )""",
    ),
    (
        "statistics_meta",
        """CREATE TABLE statistics_meta (
            id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
            statistic_id VARCHAR(255),
            source VARCHAR(32),
            unit_of_measurement VARCHAR(255),
            has_mean BOOLEAN,
            has_sum BOOLEAN,
            name VARCHAR(255)
        )""",
    ),
    (
        "statistics",
        """CREATE TABLE statistics (
            id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
            created_ts DOUBLE,
            metadata_id INTEGER,
            start_ts DOUBLE,
            mean DOUBLE, min DOUBLE, max DOUBLE,
            last_reset_ts DOUBLE, state DOUBLE, sum DOUBLE
        )""",
    ),
    (
        "statistics_short_term",
        """CREATE TABLE statistics_short_term (
            id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
            created_ts DOUBLE,
            metadata_id INTEGER,
            start_ts DOUBLE,
            mean DOUBLE, min DOUBLE, max DOUBLE,
            last_reset_ts DOUBLE, state DOUBLE, sum DOUBLE
        )""",
    ),
)

POSTGRES_TYPE_MAP = (
    ("INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY", "SERIAL PRIMARY KEY"),
    ("LONGTEXT", "TEXT"),
    ("VARBINARY(16)", "BYTEA"),
    ("DOUBLE", "DOUBLE PRECISION"),
    ("SMALLINT", "SMALLINT"),
)


def _postgres_ddl(ddl: str) -> str:
    out = ddl
    for mysql_type, pg_type in POSTGRES_TYPE_MAP:
        out = out.replace(mysql_type, pg_type)
    # Postgres declares indexes separately; drop the inline KEY clauses.
    lines = [
        line
        for line in out.splitlines()
        if not line.strip().startswith(("KEY ", "UNIQUE KEY "))
    ]
    joined = "\n".join(lines)
    return joined.replace(",\n        )", "\n        )").replace(",\n)", "\n)")


def _chunks(rows: list[Any], size: int = 500) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def load_fixture_into_mysql(sqlite_path: str | Path, engine: Engine) -> dict[str, int]:
    """Recreate the fixture inside *engine* and return per-table row counts."""
    dialect = engine.dialect.name
    source = sqlite3.connect(str(sqlite_path))
    source.row_factory = sqlite3.Row
    counts: dict[str, int] = {}

    with engine.begin() as conn:
        for table, _ in reversed(MYSQL_DDL):
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        for _table, ddl in MYSQL_DDL:
            conn.execute(text(_postgres_ddl(ddl) if dialect == "postgresql" else ddl))

    for table, _ in MYSQL_DDL:
        rows = [dict(row) for row in source.execute(f"SELECT * FROM {table}")]
        counts[table] = len(rows)
        if not rows:
            continue
        columns = list(rows[0].keys())
        placeholders = ", ".join(f":{c}" for c in columns)
        statement = text(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        )
        with engine.begin() as conn:
            for chunk in _chunks(rows):
                conn.execute(statement, chunk)
        _LOGGER.info("Loaded %d rows into %s", len(rows), table)

    source.close()
    return counts
