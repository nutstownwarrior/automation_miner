"""SQL against the recorder, written once for SQLite / MariaDB / PostgreSQL.

The post-2023 normalised schema is the primary target:

* ``states`` JOIN ``states_meta`` ON ``metadata_id`` gives ``entity_id``,
* LEFT JOIN ``state_attributes`` ON ``attributes_id`` gives the JSON attributes,
* ``old_state_id`` links to the previous row (the ``from`` state),
* context is stored *binary* (``context_id_bin`` etc.) as 16-byte ULIDs,
* ``events`` JOIN ``event_types`` / ``event_data`` mirrors the same shape.

Older schemas (text context columns, ISO timestamps, no ``states_meta``) are
detected by :mod:`amminer.discovery.recorder` and handled here too.
"""

from __future__ import annotations

import binascii
import json
import logging
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from .models import RecorderEvent, StateChange

_LOGGER = logging.getLogger(__name__)

#: Events that establish an automation/script origin for a context chain.
ORIGIN_EVENT_TYPES = ("automation_triggered", "script_started")

#: Recorder drops attributes larger than this; we never rely on big blobs.
MAX_ATTRIBUTE_BYTES = 16 * 1024


def decode_context(value: Any) -> str | None:
    """Normalise a context id from either the binary or the text column."""
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            return None
        return binascii.hexlify(raw).decode("ascii")
    return str(value)


def _decode_attributes(payload: Any) -> dict[str, Any]:
    if not payload:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError:
            return {}
    if len(payload) > MAX_ATTRIBUTE_BYTES:
        # The recorder itself caps attribute size; anything larger is suspect.
        _LOGGER.debug("Skipping oversized attribute blob (%d bytes)", len(payload))
        return {}
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


class RecorderQueries:
    """Dialect-aware queries against one recorder database."""

    def __init__(self, engine: Engine, info) -> None:
        self.engine = engine
        self.info = info

    # ------------------------------------------------------------------
    @property
    def _context_columns(self) -> str:
        if self.info.has_binary_context:
            return "s.context_id_bin AS ctx, s.context_user_id_bin AS ctx_user, s.context_parent_id_bin AS ctx_parent"
        if self.info.has_text_context:
            return "s.context_id AS ctx, s.context_user_id AS ctx_user, s.context_parent_id AS ctx_parent"
        return "NULL AS ctx, NULL AS ctx_user, NULL AS ctx_parent"

    @property
    def _event_context_columns(self) -> str:
        if self.info.has_binary_context:
            return "e.context_id_bin AS ctx, e.context_user_id_bin AS ctx_user, e.context_parent_id_bin AS ctx_parent"
        if self.info.has_text_context:
            return "e.context_id AS ctx, e.context_user_id AS ctx_user, e.context_parent_id AS ctx_parent"
        return "NULL AS ctx, NULL AS ctx_user, NULL AS ctx_parent"

    def _states_sql(self, entity_filter: bool, with_attributes: bool) -> str:
        if self.info.normalised_schema:
            select_entity = "sm.entity_id AS entity_id"
            join_meta = "JOIN states_meta sm ON s.metadata_id = sm.metadata_id"
            where_entity = "AND sm.entity_id IN :entity_ids" if entity_filter else ""
        else:
            select_entity = "s.entity_id AS entity_id"
            join_meta = ""
            where_entity = "AND s.entity_id IN :entity_ids" if entity_filter else ""

        if with_attributes:
            attr_select = "sa.shared_attrs AS attrs"
            attr_join = "LEFT JOIN state_attributes sa ON s.attributes_id = sa.attributes_id"
        else:
            attr_select = "NULL AS attrs"
            attr_join = ""

        return f"""
            SELECT {select_entity},
                   s.state AS state,
                   s.last_updated_ts AS ts,
                   s.last_changed_ts AS changed_ts,
                   os.state AS old_state,
                   {attr_select},
                   {self._context_columns}
            FROM states s
            {join_meta}
            {attr_join}
            LEFT JOIN states os ON s.old_state_id = os.state_id
            WHERE s.last_updated_ts >= :start_ts
              AND s.last_updated_ts < :end_ts
              {where_entity}
            ORDER BY s.last_updated_ts ASC
        """

    # ------------------------------------------------------------------
    def iter_state_changes(
        self,
        start_ts: float,
        end_ts: float,
        entity_ids: Sequence[str] | None = None,
        with_attributes: bool = False,
        chunk_size: int = 20000,
    ) -> Iterator[StateChange]:
        """Stream ``StateChange`` rows in chronological order."""
        params: dict[str, Any] = {"start_ts": start_ts, "end_ts": end_ts}
        entity_filter = bool(entity_ids)
        statement = text(self._states_sql(entity_filter, with_attributes))
        if entity_filter:
            statement = statement.bindparams(
                bindparam("entity_ids", expanding=True)
            )
            params["entity_ids"] = list(entity_ids or [])

        with self.engine.connect().execution_options(
            stream_results=True, yield_per=chunk_size
        ) as conn:
            for row in conn.execute(statement, params):
                mapping = row._mapping
                state = mapping["state"]
                if state is None:
                    continue
                yield StateChange(
                    entity_id=mapping["entity_id"],
                    state=str(state),
                    ts=float(mapping["ts"] or 0.0),
                    old_state=(
                        str(mapping["old_state"]) if mapping["old_state"] is not None else None
                    ),
                    last_changed_ts=(
                        float(mapping["changed_ts"]) if mapping["changed_ts"] is not None else None
                    ),
                    attributes=_decode_attributes(mapping["attrs"]),
                    context_id=decode_context(mapping["ctx"]),
                    context_user_id=decode_context(mapping["ctx_user"]),
                    context_parent_id=decode_context(mapping["ctx_parent"]),
                )

    def state_changes(self, *args: Any, **kwargs: Any) -> list[StateChange]:
        return list(self.iter_state_changes(*args, **kwargs))

    # ------------------------------------------------------------------
    def iter_events(
        self,
        start_ts: float,
        end_ts: float,
        event_types: Iterable[str] = ORIGIN_EVENT_TYPES,
        chunk_size: int = 20000,
    ) -> Iterator[RecorderEvent]:
        """Stream ``events`` rows of the requested types."""
        types = list(event_types)
        if not types or "events" not in set(self.info.tables):
            return

        has_event_types = "event_types" in set(self.info.tables)
        if has_event_types:
            sql = f"""
                SELECT et.event_type AS event_type,
                       e.time_fired_ts AS ts,
                       ed.shared_data AS data,
                       {self._event_context_columns}
                FROM events e
                JOIN event_types et ON e.event_type_id = et.event_type_id
                LEFT JOIN event_data ed ON e.data_id = ed.data_id
                WHERE et.event_type IN :event_types
                  AND e.time_fired_ts >= :start_ts
                  AND e.time_fired_ts < :end_ts
                ORDER BY e.time_fired_ts ASC
            """
        else:
            sql = f"""
                SELECT e.event_type AS event_type,
                       e.time_fired_ts AS ts,
                       e.event_data AS data,
                       {self._event_context_columns}
                FROM events e
                WHERE e.event_type IN :event_types
                  AND e.time_fired_ts >= :start_ts
                  AND e.time_fired_ts < :end_ts
                ORDER BY e.time_fired_ts ASC
            """

        statement = text(sql).bindparams(
            bindparam("event_types", expanding=True)
        )
        params = {"event_types": types, "start_ts": start_ts, "end_ts": end_ts}
        with self.engine.connect().execution_options(
            stream_results=True, yield_per=chunk_size
        ) as conn:
            for row in conn.execute(statement, params):
                mapping = row._mapping
                yield RecorderEvent(
                    event_type=str(mapping["event_type"]),
                    ts=float(mapping["ts"] or 0.0),
                    data=_decode_attributes(mapping["data"]),
                    context_id=decode_context(mapping["ctx"]),
                    context_user_id=decode_context(mapping["ctx_user"]),
                    context_parent_id=decode_context(mapping["ctx_parent"]),
                )

    def events(self, *args: Any, **kwargs: Any) -> list[RecorderEvent]:
        return list(self.iter_events(*args, **kwargs))

    # ------------------------------------------------------------------
    def entity_ids(self) -> list[str]:
        """Every entity the recorder has ever stored."""
        if self.info.normalised_schema:
            sql = "SELECT entity_id FROM states_meta"
        else:
            sql = "SELECT DISTINCT entity_id FROM states"
        with self.engine.connect() as conn:
            return [row[0] for row in conn.execute(text(sql)) if row[0]]

    def time_bounds(self) -> tuple[float | None, float | None]:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT MIN(last_updated_ts), MAX(last_updated_ts) FROM states")
            ).one_or_none()
        if not row or row[0] is None:
            return None, None
        return float(row[0]), float(row[1])

    # --- long-term statistics ----------------------------------------
    def statistics(
        self,
        statistic_ids: Sequence[str] | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
        short_term: bool = False,
    ) -> list[dict[str, Any]]:
        """Read long-term statistics.

        LTS survives ``purge_keep_days``, so it is the only usable source of
        numeric seasonality on a default 10-day-retention install.
        """
        table = "statistics_short_term" if short_term else "statistics"
        if table not in set(self.info.tables) or "statistics_meta" not in set(self.info.tables):
            return []
        clauses = ["1=1"]
        params: dict[str, Any] = {}
        if statistic_ids:
            clauses.append("sm.statistic_id IN :statistic_ids")
            params["statistic_ids"] = list(statistic_ids)
        if start_ts is not None:
            clauses.append("st.start_ts >= :start_ts")
            params["start_ts"] = start_ts
        if end_ts is not None:
            clauses.append("st.start_ts < :end_ts")
            params["end_ts"] = end_ts

        sql = f"""
            SELECT sm.statistic_id AS statistic_id,
                   sm.unit_of_measurement AS unit,
                   st.start_ts AS start_ts,
                   st.mean AS mean, st.min AS min, st.max AS max, st.sum AS sum
            FROM {table} st
            JOIN statistics_meta sm ON st.metadata_id = sm.id
            WHERE {' AND '.join(clauses)}
            ORDER BY st.start_ts ASC
        """
        statement = text(sql)
        if statistic_ids:
            statement = statement.bindparams(
                bindparam("statistic_ids", expanding=True)
            )
        with self.engine.connect() as conn:
            return [dict(row._mapping) for row in conn.execute(statement, params)]

    def statistic_ids(self) -> list[str]:
        if "statistics_meta" not in set(self.info.tables):
            return []
        with self.engine.connect() as conn:
            return [
                row[0]
                for row in conn.execute(text("SELECT statistic_id FROM statistics_meta"))
                if row[0]
            ]
