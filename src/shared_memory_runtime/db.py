"""SQLite schema and projection primitives for the derived Runtime database."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from .models import ExperienceRecord, FeedbackEvent


class DatabaseCorruptionError(sqlite3.DatabaseError):
    """The disposable projection cannot be safely opened or validated."""

    def __init__(self, database_path: Path, detail: str):
        self.database_path = Path(database_path)
        super().__init__(f"SQLite projection is corrupt: {self.database_path}: {detail}")


REQUIRED_TABLES = {
    "experiences",
    "experience_anchors",
    "experience_feedback",
    "experience_fts",
}


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences(
  id TEXT PRIMARY KEY,
  task_id TEXT,
  scope TEXT NOT NULL CHECK(scope IN ('global', 'project')),
  project_key TEXT NULL,
  kind TEXT NOT NULL,
  fragment_status TEXT NOT NULL,
  experience_verification TEXT NOT NULL,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  capsule TEXT NOT NULL,
  confidence TEXT NOT NULL,
  importance INTEGER NOT NULL CHECK(importance BETWEEN 1 AND 5),
  source_path TEXT NOT NULL,
  source_hash TEXT NOT NULL,
  remote_verified INTEGER NOT NULL CHECK(remote_verified IN (0, 1)),
  created_at TEXT NOT NULL,
  updated_at TEXT,
  CHECK((scope = 'project' AND project_key IS NOT NULL) OR
        (scope = 'global' AND project_key IS NULL))
);

CREATE TABLE IF NOT EXISTS experience_anchors(
  experience_id TEXT NOT NULL REFERENCES experiences(id) ON DELETE CASCADE,
  anchor_type TEXT NOT NULL,
  anchor_value TEXT NOT NULL,
  PRIMARY KEY (experience_id, anchor_type, anchor_value)
);

CREATE TABLE IF NOT EXISTS experience_feedback(
  experience_id TEXT PRIMARY KEY REFERENCES experiences(id) ON DELETE CASCADE,
  reuse_count INTEGER NOT NULL DEFAULT 0,
  success_count INTEGER NOT NULL DEFAULT 0,
  failure_count INTEGER NOT NULL DEFAULT 0,
  last_used_at TEXT,
  last_result TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS experience_fts USING fts5(
  experience_id UNINDEXED,
  title,
  summary,
  capsule,
  trigger,
  core_action,
  applicability,
  does_not_apply,
  exceptions,
  tokenize = 'unicode61'
);
"""


def open_database(database_path: Path) -> sqlite3.Connection:
    database_path = Path(database_path).expanduser()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = None
    try:
        connection = sqlite3.connect(str(database_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        initialize_schema(connection)
        connection.execute("PRAGMA journal_mode = WAL")
        return connection
    except sqlite3.DatabaseError as exc:
        if connection is not None:
            connection.close()
        raise DatabaseCorruptionError(database_path, str(exc)) from exc


def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS experience_fts_trigram "
            "USING fts5(experience_id UNINDEXED, text, tokenize='trigram')"
        )
    except sqlite3.OperationalError:
        # FTS5 is required by the schema; trigram is an optional acceleration.
        pass
    connection.commit()


def validate_database(connection: sqlite3.Connection) -> None:
    """Validate the projection after a rebuild and before it becomes active."""

    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise sqlite3.DatabaseError(f"integrity_check={integrity[0] if integrity else 'missing'}")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        missing = REQUIRED_TABLES - tables
        if missing:
            raise sqlite3.DatabaseError(f"missing tables: {sorted(missing)}")
        connection.execute("SELECT count(*) FROM experiences").fetchone()
        connection.execute("SELECT count(*) FROM experience_anchors").fetchone()
        connection.execute("SELECT count(*) FROM experience_feedback").fetchone()
        connection.execute("SELECT count(*) FROM experience_fts").fetchone()
    except sqlite3.DatabaseError as exc:
        raise DatabaseCorruptionError(Path(connection.execute("PRAGMA database_list").fetchone()[2]), str(exc)) from exc


def validate_database_file(database_path: Path) -> None:
    """Validate an existing file without creating or replacing it."""

    database_path = Path(database_path)
    if not database_path.exists():
        return
    connection = None
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        validate_database(connection)
    except sqlite3.DatabaseError as exc:
        if isinstance(exc, DatabaseCorruptionError):
            raise
        raise DatabaseCorruptionError(database_path, str(exc)) from exc
    finally:
        if connection is not None:
            connection.close()


def checkpoint(connection: sqlite3.Connection) -> None:
    """Flush the temporary WAL before its database file is installed."""

    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.DatabaseError as exc:
        database_path = Path(connection.execute("PRAGMA database_list").fetchone()[2])
        raise DatabaseCorruptionError(database_path, str(exc)) from exc


def supports_trigram(connection: sqlite3.Connection) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'experience_fts_trigram'"
        ).fetchone()
        is not None
    )


def _applicability_text(record: ExperienceRecord) -> str:
    return json.dumps(record.applicability.as_mapping(), ensure_ascii=False, sort_keys=True)


def _insert_anchors(connection: sqlite3.Connection, record: ExperienceRecord) -> None:
    anchors = set(record.anchors)
    anchors.add(("identity.canonical_id", record.canonical_id))
    anchors.add(("identity.outcome", record.outcome))
    anchors.update(("relation.supersedes", old_id) for old_id in record.supersedes)
    connection.executemany(
        "INSERT INTO experience_anchors(experience_id, anchor_type, anchor_value) VALUES (?, ?, ?)",
        ((record.id, anchor_type, value) for anchor_type, value in sorted(anchors)),
    )


def upsert_experience(
    connection: sqlite3.Connection,
    record: ExperienceRecord,
    remote_verified: bool,
) -> None:
    connection.execute(
        """
        INSERT INTO experiences(
          id, task_id, scope, project_key, kind, fragment_status,
          experience_verification, title, summary, capsule, confidence,
          importance, source_path, source_hash, remote_verified,
          created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          task_id = excluded.task_id,
          scope = excluded.scope,
          project_key = excluded.project_key,
          kind = excluded.kind,
          fragment_status = excluded.fragment_status,
          experience_verification = excluded.experience_verification,
          title = excluded.title,
          summary = excluded.summary,
          capsule = excluded.capsule,
          confidence = excluded.confidence,
          importance = excluded.importance,
          source_path = excluded.source_path,
          source_hash = excluded.source_hash,
          remote_verified = excluded.remote_verified,
          created_at = excluded.created_at,
          updated_at = excluded.updated_at
        """,
        (
            record.id,
            record.task_id,
            record.scope,
            record.project_key,
            record.kind,
            record.fragment_status,
            record.experience_verification,
            record.title,
            record.summary,
            record.capsule,
            record.confidence,
            record.importance,
            record.source_path,
            record.source_hash,
            int(remote_verified),
            record.created_at,
            record.updated_at,
        ),
    )
    connection.execute("DELETE FROM experience_anchors WHERE experience_id = ?", (record.id,))
    _insert_anchors(connection, record)
    connection.execute("DELETE FROM experience_fts WHERE experience_id = ?", (record.id,))
    connection.execute(
        """
        INSERT INTO experience_fts(
          experience_id, title, summary, capsule, trigger, core_action,
          applicability, does_not_apply, exceptions
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.id,
            record.title,
            record.summary,
            record.capsule,
            record.trigger,
            record.core_action,
            _applicability_text(record),
            record.does_not_apply,
            record.exceptions,
        ),
    )
    if supports_trigram(connection):
        connection.execute("DELETE FROM experience_fts_trigram WHERE experience_id = ?", (record.id,))
        connection.execute(
            "INSERT INTO experience_fts_trigram(experience_id, text) VALUES (?, ?)",
            (record.id, record.effective_text),
        )


def _aggregate_events(events: Iterable[FeedbackEvent]) -> Mapping[str, object]:
    grouped = defaultdict(list)
    for event in events:
        grouped[event.canonical_id].append(event)
    return grouped


def rebuild_feedback(connection: sqlite3.Connection, records: Sequence[ExperienceRecord]) -> None:
    events_by_canonical = _aggregate_events(
        record.feedback for record in records if record.feedback is not None
    )
    for record in records:
        events = events_by_canonical.get(record.id, [])
        ordered = sorted(events, key=lambda event: (event.used_at, event.experience_id))
        success_count = sum(event.result == "success" for event in events)
        failure_count = sum(event.result == "failure" for event in events)
        last_event = ordered[-1] if ordered else None
        connection.execute(
            """
            INSERT INTO experience_feedback(
              experience_id, reuse_count, success_count, failure_count,
              last_used_at, last_result
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(experience_id) DO UPDATE SET
              reuse_count = excluded.reuse_count,
              success_count = excluded.success_count,
              failure_count = excluded.failure_count,
              last_used_at = excluded.last_used_at,
              last_result = excluded.last_result
            """,
            (
                record.id,
                len(events),
                success_count,
                failure_count,
                last_event.used_at if last_event else None,
                last_event.result if last_event else None,
            ),
        )


def clear_projection(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM experience_fts")
    if supports_trigram(connection):
        connection.execute("DELETE FROM experience_fts_trigram")
    connection.execute("DELETE FROM experience_feedback")
    connection.execute("DELETE FROM experience_anchors")
    connection.execute("DELETE FROM experiences")
