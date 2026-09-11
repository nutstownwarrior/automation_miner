"""The add-on's private SQLite store.

Lives under the add-on's OWN ``/config`` (mapped via ``addon_config``), never in
Home Assistant's recorder database.  Holds discovered suggestions, dismissals,
feedback, override history, backtests, shadow-mode logs and run metadata.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 1

STATUS_NEW = "new"
STATUS_DISMISSED = "dismissed"
STATUS_ACCEPTED = "accepted"
STATUS_SHADOW = "shadow"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS suggestions (
    id             TEXT PRIMARY KEY,
    miner          TEXT NOT NULL,
    title          TEXT NOT NULL,
    summary        TEXT,
    score          REAL NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'new',
    payload        TEXT NOT NULL,
    first_seen_ts  REAL NOT NULL,
    last_seen_ts   REAL NOT NULL,
    seen_count     INTEGER NOT NULL DEFAULT 1,
    run_id         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_suggestions_status ON suggestions(status);
CREATE INDEX IF NOT EXISTS idx_suggestions_miner  ON suggestions(miner);

CREATE TABLE IF NOT EXISTS dismissals (
    suggestion_id TEXT PRIMARY KEY,
    ts            REAL NOT NULL,
    reason        TEXT,
    signature     TEXT
);
CREATE INDEX IF NOT EXISTS idx_dismissals_signature ON dismissals(signature);

CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    suggestion_id TEXT,
    kind          TEXT NOT NULL,
    ts            REAL NOT NULL,
    payload       TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_suggestion ON feedback(suggestion_id);

CREATE TABLE IF NOT EXISTS backtests (
    suggestion_id        TEXT PRIMARY KEY,
    ts                   REAL NOT NULL,
    precision_score      REAL,
    recall_score         REAL,
    true_fires           INTEGER,
    false_fires          INTEGER,
    missed               INTEGER,
    false_fires_per_week REAL,
    passed               INTEGER NOT NULL DEFAULT 0,
    payload              TEXT
);

CREATE TABLE IF NOT EXISTS overrides (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                    REAL NOT NULL,
    entity_id             TEXT NOT NULL,
    automation_entity_id  TEXT,
    automation_state      TEXT,
    human_state           TEXT,
    delay_seconds         REAL,
    UNIQUE(ts, entity_id, human_state)
);
CREATE INDEX IF NOT EXISTS idx_overrides_entity ON overrides(entity_id);
CREATE INDEX IF NOT EXISTS idx_overrides_auto   ON overrides(automation_entity_id);

CREATE TABLE IF NOT EXISTS shadow_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    suggestion_id TEXT NOT NULL,
    ts            REAL NOT NULL,
    would_fire    INTEGER NOT NULL DEFAULT 1,
    matched       INTEGER,
    payload       TEXT
);
CREATE INDEX IF NOT EXISTS idx_shadow_suggestion ON shadow_events(suggestion_id);

CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_ts   REAL NOT NULL,
    finished_ts  REAL,
    status       TEXT NOT NULL DEFAULT 'running',
    stats        TEXT,
    error        TEXT
);

CREATE TABLE IF NOT EXISTS generations (
    suggestion_id TEXT PRIMARY KEY,
    ts            REAL NOT NULL,
    digest        TEXT NOT NULL,
    source        TEXT,
    payload       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gap_suggestions (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    payload     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'new',
    first_seen_ts REAL NOT NULL,
    last_seen_ts  REAL NOT NULL
);
"""


def _json(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


class Store:
    """Small, thread-safe wrapper over the add-on's SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # ------------------------------------------------------------------
    def _migrate(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            return cursor

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params))

    # --- meta ---------------------------------------------------------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        rows = self._query("SELECT value FROM meta WHERE key = ?", (key,))
        return rows[0]["value"] if rows else default

    def set_meta(self, key: str, value: str) -> None:
        self._execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # --- runs ---------------------------------------------------------
    def start_run(self) -> int:
        cursor = self._execute("INSERT INTO runs(started_ts) VALUES(?)", (time.time(),))
        return int(cursor.lastrowid or 0)

    def finish_run(
        self, run_id: int, status: str = "ok", stats: dict | None = None, error: str | None = None
    ) -> None:
        self._execute(
            "UPDATE runs SET finished_ts = ?, status = ?, stats = ?, error = ? WHERE id = ?",
            (time.time(), status, _json(stats or {}), error, run_id),
        )

    def close_interrupted_runs(self) -> int:
        """Mark runs that never finished, so nothing sits at "running" forever.

        A run row is opened before mining and closed after persisting.  If the
        add-on is stopped in between - a Supervisor restart during the nightly
        analysis - the row is left open and the status page shows a run that has
        been in progress since whenever that was.
        """
        cursor = self._execute(
            "UPDATE runs SET status = ?, finished_ts = ?, error = ?"
            " WHERE finished_ts IS NULL AND status = 'running'",
            ("interrupted", time.time(), "the add-on stopped while this run was in progress"),
        )
        return cursor.rowcount or 0

    def last_run(self) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM runs ORDER BY id DESC LIMIT 1")
        if not rows:
            return None
        run = dict(rows[0])
        run["stats"] = json.loads(run["stats"]) if run.get("stats") else {}
        return run

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
        out = []
        for row in rows:
            run = dict(row)
            run["stats"] = json.loads(run["stats"]) if run.get("stats") else {}
            out.append(run)
        return out

    # --- suggestions --------------------------------------------------
    def upsert_suggestion(
        self,
        suggestion_id: str,
        miner: str,
        title: str,
        summary: str,
        score: float,
        payload: dict[str, Any],
        run_id: int | None = None,
    ) -> str:
        """Insert or refresh a suggestion; never resurrects a dismissed one."""
        now = time.time()
        with self._lock:
            existing = self._conn.execute(
                "SELECT status FROM suggestions WHERE id = ?", (suggestion_id,)
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO suggestions(id, miner, title, summary, score, status, payload,"
                    " first_seen_ts, last_seen_ts, seen_count, run_id)"
                    " VALUES(?,?,?,?,?,?,?,?,?,1,?)",
                    (
                        suggestion_id,
                        miner,
                        title,
                        summary,
                        float(score),
                        STATUS_NEW,
                        _json(payload),
                        now,
                        now,
                        run_id,
                    ),
                )
                status = STATUS_NEW
            else:
                status = existing["status"]
                # A dismissed suggestion keeps its status: dismissals are sticky.
                self._conn.execute(
                    "UPDATE suggestions SET miner=?, title=?, summary=?, score=?, payload=?,"
                    " last_seen_ts=?, seen_count=seen_count+1, run_id=? WHERE id=?",
                    (
                        miner,
                        title,
                        summary,
                        float(score),
                        _json(payload),
                        now,
                        run_id,
                        suggestion_id,
                    ),
                )
            self._conn.commit()
        return status

    def get_suggestion(self, suggestion_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM suggestions WHERE id = ?", (suggestion_id,))
        return self._row_to_suggestion(rows[0]) if rows else None

    def _row_to_suggestion(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        try:
            data["payload"] = json.loads(data["payload"])
        except (json.JSONDecodeError, TypeError):
            data["payload"] = {}
        backtest = self._query(
            "SELECT * FROM backtests WHERE suggestion_id = ?", (data["id"],)
        )
        if backtest:
            bt = dict(backtest[0])
            try:
                bt["payload"] = json.loads(bt["payload"]) if bt.get("payload") else {}
            except (json.JSONDecodeError, TypeError):
                bt["payload"] = {}
            data["backtest"] = bt
        return data

    def list_suggestions(
        self,
        status: str | Iterable[str] | None = None,
        miner: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses, params = ["1=1"], []
        if status:
            statuses = [status] if isinstance(status, str) else list(status)
            clauses.append(f"status IN ({','.join('?' * len(statuses))})")
            params.extend(statuses)
        if miner:
            clauses.append("miner = ?")
            params.append(miner)
        params.append(limit)
        rows = self._query(
            f"SELECT * FROM suggestions WHERE {' AND '.join(clauses)}"
            " ORDER BY score DESC, last_seen_ts DESC LIMIT ?",
            params,
        )
        return [self._row_to_suggestion(row) for row in rows]

    def set_status(self, suggestion_id: str, status: str) -> None:
        self._execute("UPDATE suggestions SET status = ? WHERE id = ?", (status, suggestion_id))

    def prune_suggestions(self, run_id: int) -> int:
        """Drop untouched ``new`` suggestions from earlier runs."""
        cursor = self._execute(
            "DELETE FROM suggestions WHERE status = ? AND (run_id IS NULL OR run_id < ?)",
            (STATUS_NEW, run_id),
        )
        return cursor.rowcount or 0

    # --- dismissals & feedback ---------------------------------------
    def dismiss(self, suggestion_id: str, reason: str | None = None, signature: str | None = None) -> None:
        self._execute(
            "INSERT INTO dismissals(suggestion_id, ts, reason, signature) VALUES(?,?,?,?)"
            " ON CONFLICT(suggestion_id) DO UPDATE SET ts=excluded.ts, reason=excluded.reason",
            (suggestion_id, time.time(), reason, signature),
        )
        self.set_status(suggestion_id, STATUS_DISMISSED)
        self.add_feedback(suggestion_id, "dismissed", {"reason": reason})

    def restore(self, suggestion_id: str) -> bool:
        """Undo a dismissal, including the record the next run consults.

        Setting the status back to ``new`` on its own is not a restore: the
        dismissals table is what mining filters against, so the suggestion is
        dropped again on the next run and then pruned.  The user sees it
        reappear and then quietly disappear.
        """
        row = self._query("SELECT 1 FROM suggestions WHERE id = ?", (suggestion_id,))
        if not row:
            return False
        signatures = self._query(
            "SELECT signature FROM dismissals WHERE suggestion_id = ?", (suggestion_id,)
        )
        self._execute("DELETE FROM dismissals WHERE suggestion_id = ?", (suggestion_id,))
        for entry in signatures:
            if entry["signature"]:
                self._execute(
                    "DELETE FROM dismissals WHERE signature = ?", (entry["signature"],)
                )
        self.set_status(suggestion_id, STATUS_NEW)
        self.add_feedback(suggestion_id, "restored")
        return True

    def is_dismissed(self, suggestion_id: str, signature: str | None = None) -> bool:
        rows = self._query("SELECT 1 FROM dismissals WHERE suggestion_id = ?", (suggestion_id,))
        if rows:
            return True
        if signature:
            rows = self._query("SELECT 1 FROM dismissals WHERE signature = ?", (signature,))
            return bool(rows)
        return False

    def dismissed_signatures(self) -> set[str]:
        return {
            row["signature"]
            for row in self._query("SELECT signature FROM dismissals WHERE signature IS NOT NULL")
        }

    def dismissed_ids(self) -> set[str]:
        return {row["suggestion_id"] for row in self._query("SELECT suggestion_id FROM dismissals")}

    def add_feedback(self, suggestion_id: str | None, kind: str, payload: dict | None = None) -> None:
        self._execute(
            "INSERT INTO feedback(suggestion_id, kind, ts, payload) VALUES(?,?,?,?)",
            (suggestion_id, kind, time.time(), _json(payload or {})),
        )

    def feedback_for(self, suggestion_id: str) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM feedback WHERE suggestion_id = ? ORDER BY ts DESC", (suggestion_id,)
        )
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item["payload"]) if item.get("payload") else {}
            except (json.JSONDecodeError, TypeError):
                item["payload"] = {}
            out.append(item)
        return out

    # --- backtests ----------------------------------------------------
    def save_backtest(self, suggestion_id: str, result: dict[str, Any]) -> None:
        self._execute(
            "INSERT INTO backtests(suggestion_id, ts, precision_score, recall_score, true_fires,"
            " false_fires, missed, false_fires_per_week, passed, payload)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(suggestion_id) DO UPDATE SET ts=excluded.ts,"
            " precision_score=excluded.precision_score, recall_score=excluded.recall_score,"
            " true_fires=excluded.true_fires, false_fires=excluded.false_fires,"
            " missed=excluded.missed, false_fires_per_week=excluded.false_fires_per_week,"
            " passed=excluded.passed, payload=excluded.payload",
            (
                suggestion_id,
                time.time(),
                result.get("precision"),
                result.get("recall"),
                result.get("true_fires"),
                result.get("false_fires"),
                result.get("missed"),
                result.get("false_fires_per_week"),
                1 if result.get("passed") else 0,
                _json(result),
            ),
        )

    def get_backtest(self, suggestion_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM backtests WHERE suggestion_id = ?", (suggestion_id,))
        if not rows:
            return None
        data = dict(rows[0])
        try:
            data["payload"] = json.loads(data["payload"]) if data.get("payload") else {}
        except (json.JSONDecodeError, TypeError):
            data["payload"] = {}
        return data

    # --- overrides ----------------------------------------------------
    def record_overrides(self, overrides: Iterable[Any]) -> int:
        inserted = 0
        with self._lock:
            for override in overrides:
                data = override.as_dict() if hasattr(override, "as_dict") else dict(override)
                try:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO overrides(ts, entity_id, automation_entity_id,"
                        " automation_state, human_state, delay_seconds) VALUES(?,?,?,?,?,?)",
                        (
                            data.get("ts"),
                            data.get("entity_id"),
                            data.get("automation_entity_id"),
                            data.get("automation_state"),
                            data.get("human_state"),
                            data.get("delay_seconds"),
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    continue
            self._conn.commit()
        return inserted

    def overrides_for_automation(self, automation_entity_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._query(
                "SELECT * FROM overrides WHERE automation_entity_id = ? ORDER BY ts DESC",
                (automation_entity_id,),
            )
        ]

    def override_counts(self) -> dict[str, int]:
        rows = self._query(
            "SELECT automation_entity_id AS a, COUNT(*) AS c FROM overrides"
            " WHERE automation_entity_id IS NOT NULL GROUP BY automation_entity_id"
        )
        return {row["a"]: int(row["c"]) for row in rows}

    def recent_overrides(self, limit: int = 100) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._query("SELECT * FROM overrides ORDER BY ts DESC LIMIT ?", (limit,))
        ]

    # --- shadow mode --------------------------------------------------
    def log_shadow_fire(
        self, suggestion_id: str, ts: float, matched: bool | None = None, payload: dict | None = None
    ) -> None:
        self._execute(
            "INSERT INTO shadow_events(suggestion_id, ts, would_fire, matched, payload)"
            " VALUES(?,?,1,?,?)",
            (suggestion_id, ts, None if matched is None else int(matched), _json(payload or {})),
        )

    def last_shadow_ts(self, suggestion_id: str) -> float:
        """Newest fire already recorded, so a re-run does not double-count."""
        rows = self._query(
            "SELECT MAX(ts) AS newest FROM shadow_events WHERE suggestion_id = ?",
            (suggestion_id,),
        )
        newest = rows[0]["newest"] if rows else None
        return float(newest) if newest is not None else 0.0

    def shadow_report(self, suggestion_id: str) -> dict[str, Any]:
        rows = self._query(
            "SELECT matched FROM shadow_events WHERE suggestion_id = ?", (suggestion_id,)
        )
        total = len(rows)
        matched = sum(1 for row in rows if row["matched"] == 1)
        unmatched = sum(1 for row in rows if row["matched"] == 0)
        return {
            "fires": total,
            "matched": matched,
            "unmatched": unmatched,
            "precision": (matched / (matched + unmatched)) if (matched + unmatched) else None,
        }

    # --- generated automations ----------------------------------------
    def save_generation(self, suggestion_id: str, digest: str, source: str, payload: dict) -> None:
        """Remember the exact automation a user was shown.

        Apply must write what was reviewed. Regenerating at apply time asks a
        non-deterministic model the same question twice and writes the second
        answer, which is not the one that was consented to.
        """
        self._execute(
            "INSERT INTO generations(suggestion_id, ts, digest, source, payload)"
            " VALUES(?,?,?,?,?)"
            " ON CONFLICT(suggestion_id) DO UPDATE SET ts=excluded.ts,"
            " digest=excluded.digest, source=excluded.source, payload=excluded.payload",
            (suggestion_id, time.time(), digest, source, _json(payload)),
        )

    def get_generation(self, suggestion_id: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT * FROM generations WHERE suggestion_id = ?", (suggestion_id,)
        )
        if not rows:
            return None
        data = dict(rows[0])
        try:
            data["payload"] = json.loads(data["payload"])
        except (json.JSONDecodeError, TypeError):
            return None
        return data

    # --- gap suggestions ---------------------------------------------
    def upsert_gap(self, gap_id: str, kind: str, title: str, payload: dict[str, Any]) -> None:
        now = time.time()
        self._execute(
            "INSERT INTO gap_suggestions(id, kind, title, payload, status, first_seen_ts, last_seen_ts)"
            " VALUES(?,?,?,?,'new',?,?)"
            " ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, title=excluded.title,"
            " last_seen_ts=excluded.last_seen_ts",
            (gap_id, kind, title, _json(payload), now, now),
        )

    def list_gaps(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._query(
                "SELECT * FROM gap_suggestions WHERE status = ? ORDER BY last_seen_ts DESC",
                (status,),
            )
        else:
            rows = self._query("SELECT * FROM gap_suggestions ORDER BY last_seen_ts DESC")
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item["payload"])
            except (json.JSONDecodeError, TypeError):
                item["payload"] = {}
            out.append(item)
        return out

    def set_gap_status(self, gap_id: str, status: str) -> None:
        self._execute("UPDATE gap_suggestions SET status = ? WHERE id = ?", (status, gap_id))

    # --- misc ---------------------------------------------------------
    def counts(self) -> dict[str, int]:
        out = {}
        for table in (
            "suggestions",
            "dismissals",
            "feedback",
            "backtests",
            "overrides",
            "shadow_events",
            "runs",
            "generations",
            "gap_suggestions",
        ):
            rows = self._query(f"SELECT COUNT(*) AS c FROM {table}")
            out[table] = int(rows[0]["c"]) if rows else 0
        return out
