"""Rebuild and incremental projection from authoritative Markdown/Git."""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .db import (
    DatabaseCorruptionError,
    checkpoint,
    clear_projection,
    open_database,
    rebuild_feedback,
    upsert_experience,
    validate_database,
    validate_database_file,
)
from .git_proof import GitEvidence, RemoteProof
from .markdown import FragmentValidationError, parse_experience_fragment, parse_front_matter, project_key
from .models import ExperienceRecord
from .recall import RecallCandidate, RecallContext, RecallRun, recall_with_stats


def default_database_path(codex_home: Optional[Path] = None) -> Path:
    configured_home = codex_home or os.environ.get("CODEX_HOME")
    root = Path(configured_home).expanduser() if configured_home else Path.home() / ".codex"
    return root / ".state" / "experience.db"


@dataclass(frozen=True)
class SkippedSource:
    source_path: str
    reason_code: str
    detail: str
    experience_id: Optional[str] = None
    scope: Optional[str] = None
    canonical_project_key: Optional[str] = None
    source_hash: Optional[str] = None
    eligibility: str = "excluded"

    def as_mapping(self) -> dict[str, Optional[str]]:
        return {
            "source_path": self.source_path,
            "experience_id": self.experience_id,
            "scope": self.scope,
            "canonical_project_key": self.canonical_project_key,
            "source_hash": self.source_hash,
            "eligibility": self.eligibility,
            "reason_code": self.reason_code,
        }


@dataclass
class RebuildReport:
    projected: List[ExperienceRecord] = field(default_factory=list)
    proofs: List[RemoteProof] = field(default_factory=list)
    skipped: List[SkippedSource] = field(default_factory=list)
    corruption_recovered: bool = False
    quarantine_paths: Tuple[str, ...] = ()

    @property
    def remote_verified_count(self) -> int:
        return sum(proof.verified for proof in self.proofs)


@dataclass(frozen=True)
class ProjectionFreshness:
    """Read-only compatibility result for the local derived projection."""

    ready: bool
    reason_code: str
    detail: str = ""
    diagnostics: Tuple[SkippedSource, ...] = ()


class ExperienceProjector:
    """Project Experience-bearing Fragments from a Shared Knowledge checkout."""

    def __init__(
        self,
        source_root: Path,
        database_path: Optional[Path] = None,
        remote_snapshot: Optional[Tuple[str, str]] = None,
    ):
        self.source_root = Path(source_root).resolve()
        self.database_path = Path(database_path) if database_path else default_database_path()
        self.git = GitEvidence(self.source_root, remote_snapshot=remote_snapshot)
        self._database_lock = threading.RLock()
        self._connections: dict[int, sqlite3.Connection] = {}
        self._recovering = False

    def _source_paths(self) -> List[Path]:
        memory_root = self.source_root / ".memory"
        if not memory_root.is_dir():
            return []
        paths = []
        for path in memory_root.rglob("*.md"):
            relative = path.relative_to(self.source_root).as_posix()
            if relative == ".memory/README.md" or relative.startswith(".memory/schema/"):
                continue
            paths.append(path)
        return sorted(paths)

    def _source_diagnostic(
        self,
        path: Path,
        source_hash: Optional[str],
        reason_code: str,
        detail: str,
    ) -> SkippedSource:
        relative = path.relative_to(self.source_root).as_posix()
        experience_id: Optional[str] = None
        scope: Optional[str] = None
        canonical_project_key: Optional[str] = None
        try:
            metadata, _ = parse_front_matter(path.read_text(encoding="utf-8", errors="replace"))
        except (FragmentValidationError, OSError, UnicodeError, ValueError):
            metadata = {}
        raw_id = metadata.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            experience_id = raw_id.strip()
        raw_scope = metadata.get("scope")
        if isinstance(raw_scope, str) and raw_scope.strip():
            scope = raw_scope.strip()
        identity = metadata.get("project")
        if scope == "project" and isinstance(identity, str) and identity.strip():
            try:
                canonical_project_key = project_key(identity)
            except FragmentValidationError:
                pass
        return SkippedSource(
            relative,
            reason_code,
            detail,
            experience_id=experience_id,
            scope=scope,
            canonical_project_key=canonical_project_key,
            source_hash=source_hash,
        )

    def _resolve_source_path(
        self, path: Path
    ) -> Tuple[Optional[ExperienceRecord], Optional[SkippedSource]]:
        relative = path.relative_to(self.source_root).as_posix()
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        marker = raw_text.find("\n---\n", 4) if raw_text.startswith("---\n") else -1
        if marker < 0 or not re.search(r"^experience:\s*$", raw_text[4:marker], re.MULTILINE):
            return None, None
        source_hash = self.git.tracked_source_hash(relative)
        if not source_hash:
            return None, self._source_diagnostic(
                path, None, "source_not_tracked", "source is not a tracked Git path"
            )
        try:
            record = parse_experience_fragment(path, self.source_root, source_hash)
        except (FragmentValidationError, OSError, UnicodeError, ValueError) as exc:
            # Non-Experience and legacy Fragments are expected in the source tree.
            message = str(exc)
            if (
                "missing Experience overlay" in message
                or "unsupported Fragment type" in message
                or ("invalid YAML front matter" in message and "\nexperience:" not in raw_text)
                or ("requires task_id" in message and "\ntask_id:" not in raw_text)
            ):
                return None, None
            reason_code = getattr(exc, "reason_code", None) or "invalid_experience"
            return None, self._source_diagnostic(path, source_hash, reason_code, message)
        return record, None

    def _parse_path(self, path: Path) -> Tuple[Optional[ExperienceRecord], Optional[RemoteProof], Optional[SkippedSource]]:
        record, skipped = self._resolve_source_path(path)
        if skipped or record is None:
            return None, None, skipped
        proof = self.git.prove_remote_persistence(record.source_path, record.source_hash)
        return record, proof, None

    def _collect_eligible_records(self) -> Tuple[List[ExperienceRecord], Tuple[SkippedSource, ...]]:
        """Resolve one authoritative eligible source set without Remote Proof."""

        records: List[ExperienceRecord] = []
        skipped: List[SkippedSource] = []
        for path in self._source_paths():
            record, diagnostic = self._resolve_source_path(path)
            if diagnostic:
                skipped.append(diagnostic)
            if record is not None:
                records.append(record)

        record_ids = {record.id for record in records}
        eligible: List[ExperienceRecord] = []
        for record in records:
            if record.outcome == "REINFORCE" and record.canonical_id not in record_ids:
                skipped.append(
                    SkippedSource(
                        record.source_path,
                        "canonical_experience_unavailable",
                        f"REINFORCE canonical_id {record.canonical_id} is not present in the authoritative scan",
                        experience_id=record.id,
                        scope=record.scope,
                        canonical_project_key=record.project_key,
                        source_hash=record.source_hash,
                    )
                )
                continue
            missing_supersedes = [old_id for old_id in record.supersedes if old_id not in record_ids]
            if record.outcome == "CORRECT" and missing_supersedes:
                skipped.append(
                    SkippedSource(
                        record.source_path,
                        "superseded_experience_unavailable",
                        f"CORRECT supersedes unavailable Experience ids: {', '.join(sorted(missing_supersedes))}",
                        experience_id=record.id,
                        scope=record.scope,
                        canonical_project_key=record.project_key,
                        source_hash=record.source_hash,
                    )
                )
                continue
            eligible.append(record)
        return eligible, tuple(sorted(skipped, key=lambda item: item.source_path))

    def _collect_projection(self) -> RebuildReport:
        eligible, skipped = self._collect_eligible_records()
        report = RebuildReport(skipped=list(skipped))
        for record in eligible:
            report.projected.append(record)
            report.proofs.append(self.git.prove_remote_persistence(record.source_path, record.source_hash))
        return report

    @contextmanager
    def _connection(self, database_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
        connection = open_database(database_path or self.database_path)
        self._connections[id(connection)] = connection
        try:
            yield connection
        finally:
            self._connections.pop(id(connection), None)
            connection.close()

    def _close_all_connections(self) -> None:
        for connection in list(self._connections.values()):
            try:
                connection.close()
            finally:
                self._connections.pop(id(connection), None)

    def _active_is_corrupt(self) -> bool:
        sidecars = (
            Path(str(self.database_path) + "-wal"),
            Path(str(self.database_path) + "-shm"),
        )
        if not self.database_path.exists():
            return any(sidecar.exists() for sidecar in sidecars)
        try:
            validate_database_file(self.database_path)
        except DatabaseCorruptionError:
            return True
        return False

    def _quarantine_active_database(self) -> Tuple[str, ...]:
        existing = [
            candidate
            for candidate in (
                self.database_path,
                Path(str(self.database_path) + "-wal"),
                Path(str(self.database_path) + "-shm"),
            )
            if candidate.exists()
        ]
        if not existing:
            return ()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
        quarantine_paths = []
        for candidate in existing:
            if candidate == self.database_path:
                target = self.database_path.with_name(f"{self.database_path.name}.corrupt-{stamp}")
            else:
                target = candidate.with_name(f"{candidate.name}.corrupt-{stamp}")
            os.replace(candidate, target)
            quarantine_paths.append(str(target))
        return tuple(quarantine_paths)

    def _prune_quarantine(self, keep: int = 3) -> None:
        groups = {}
        pattern = re.compile(r"\.corrupt-(\d{8}T\d{12}Z-[0-9a-f]+)$")
        for candidate in self.database_path.parent.glob(f"{self.database_path.name}*.corrupt-*"):
            match = pattern.search(candidate.name)
            if match:
                groups.setdefault(match.group(1), []).append(candidate)
        for stamp in sorted(groups, reverse=True)[keep:]:
            for candidate in groups[stamp]:
                candidate.unlink(missing_ok=True)

    def _validate_supersedes_graph(self, connection: sqlite3.Connection) -> None:
        ids = {row[0] for row in connection.execute("SELECT id FROM experiences")}
        for row in connection.execute(
            "SELECT anchor_value FROM experience_anchors WHERE anchor_type = 'relation.supersedes'"
        ):
            if row[0] not in ids:
                raise DatabaseCorruptionError(self.database_path, f"missing superseded Experience {row[0]}")

    def _validate_projection_file(self, database_path: Path, report: RebuildReport) -> None:
        validate_database_file(database_path)
        connection = sqlite3.connect(str(database_path))
        connection.row_factory = sqlite3.Row
        try:
            expected_ids = {record.id for record in report.projected}
            actual_ids = {row[0] for row in connection.execute("SELECT id FROM experiences")}
            if actual_ids != expected_ids:
                raise DatabaseCorruptionError(
                    database_path,
                    f"projection ids differ: expected {len(expected_ids)}, got {len(actual_ids)}",
                )
            self._validate_supersedes_graph(connection)
            connection.execute(
                "SELECT rowid FROM experience_fts WHERE experience_fts MATCH 'experience'"
            ).fetchone()
        finally:
            connection.close()

    def _active_smoke_verify(self, report: RebuildReport) -> None:
        with self._connection() as connection:
            validate_database(connection)
            self._validate_supersedes_graph(connection)
            for record, proof in zip(report.projected, report.proofs):
                if not proof.verified:
                    continue
                context = RecallContext(
                    project_key=record.project_key,
                    task_kind=record.kind,
                    anchors=record.anchor_values(),
                    query=record.title,
                )
                recall_with_stats(connection, context)

    def _build_projection(self, report: RebuildReport) -> Tuple[Path, Path]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(tempfile.mkdtemp(prefix=f"{self.database_path.name}.rebuild-", dir=self.database_path.parent))
        temporary_database = temporary_dir / self.database_path.name
        try:
            with self._connection(temporary_database) as connection:
                with connection:
                    clear_projection(connection)
                    for record, proof in zip(report.projected, report.proofs):
                        upsert_experience(connection, record, proof.verified)
                    rebuild_feedback(connection, report.projected)
                checkpoint(connection)
            self._validate_projection_file(temporary_database, report)
            return temporary_dir, temporary_database
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

    def _rebuild_locked(self) -> RebuildReport:
        self._recovering = True
        quarantine_paths = ()
        corruption_recovered = self._active_is_corrupt()
        try:
            self._close_all_connections()
            if corruption_recovered:
                quarantine_paths = self._quarantine_active_database()
            report = self._collect_projection()
            temporary_dir, temporary_database = self._build_projection(report)
            try:
                if self.database_path.exists():
                    with self._connection() as connection:
                        checkpoint(connection)
                    for sidecar in (
                        Path(str(self.database_path) + "-wal"),
                        Path(str(self.database_path) + "-shm"),
                    ):
                        sidecar.unlink(missing_ok=True)
                os.replace(temporary_database, self.database_path)
            finally:
                shutil.rmtree(temporary_dir, ignore_errors=True)
            verified_report = RebuildReport(
                projected=report.projected,
                proofs=report.proofs,
                skipped=report.skipped,
                corruption_recovered=corruption_recovered,
                quarantine_paths=quarantine_paths,
            )
            self._active_smoke_verify(verified_report)
            self._prune_quarantine()
            return verified_report
        except Exception:
            raise
        finally:
            self._recovering = False

    def rebuild(self) -> RebuildReport:
        with self._database_lock:
            return self._rebuild_locked()

    def projection_freshness(self, *, shared_knowledge_fresh: bool = False) -> ProjectionFreshness:
        """Check DB/source compatibility after formal Shared Knowledge freshness."""

        with self._database_lock:
            if not shared_knowledge_fresh:
                return ProjectionFreshness(False, "shared_knowledge_freshness_required")
            if not self.database_path.exists():
                return ProjectionFreshness(False, "projection_missing")
            try:
                validate_database_file(self.database_path)
                records, diagnostics = self._collect_eligible_records()
                connection = sqlite3.connect(str(self.database_path))
                connection.row_factory = sqlite3.Row
                try:
                    actual = {
                        (
                            row["id"],
                            row["source_path"],
                            row["scope"],
                            row["project_key"],
                            row["source_hash"],
                        )
                        for row in connection.execute(
                            "SELECT id, source_path, scope, project_key, source_hash FROM experiences"
                        )
                    }
                finally:
                    connection.close()
                expected = {
                    (
                        record.id,
                        record.source_path,
                        record.scope,
                        record.project_key,
                        record.source_hash,
                    )
                    for record in records
                }
                if actual != expected:
                    return ProjectionFreshness(False, "projection_source_mismatch", diagnostics=diagnostics)
                detail = ", ".join(
                    f"{item.source_path}:{item.reason_code}" for item in diagnostics
                )
                return ProjectionFreshness(True, "fresh", detail, diagnostics)
            except (DatabaseCorruptionError, OSError, sqlite3.DatabaseError) as exc:
                return ProjectionFreshness(False, "projection_invalid", str(exc))

    def project_path(self, source_path: str) -> RemoteProof:
        """Project one source after Remote Verify; failed proof remains ineligible."""
        with self._database_lock:
            path = self.source_root / source_path
            source_hash = self.git.tracked_source_hash(source_path)
            if not source_hash:
                return RemoteProof(False, source_path, None, None, None, None, "source_not_tracked")
            record, skipped = self._resolve_source_path(path)
            if skipped or record is None:
                if skipped:
                    raise FragmentValidationError(skipped.detail, reason_code=skipped.reason_code)
                raise FragmentValidationError("source has no Experience overlay")
            proof = self.git.prove_remote_persistence(record.source_path, record.source_hash)
            records, _ = self._collect_eligible_records()
            if not any(item.id == record.id for item in records):
                records.append(record)
            record_ids = {item.id for item in records}
            if record.outcome == "REINFORCE" and record.canonical_id not in record_ids:
                raise ValueError(
                    f"REINFORCE canonical_id {record.canonical_id} is not present in the authoritative scan"
                )
            try:
                with self._connection() as connection:
                    with connection:
                        upsert_experience(connection, record, proof.verified)
                        rebuild_feedback(connection, records)
            except (DatabaseCorruptionError, sqlite3.DatabaseError):
                self._rebuild_locked()
            return proof

    def recall(
        self, context: RecallContext, *, shared_knowledge_fresh: bool = False
    ) -> List[RecallCandidate]:
        return list(
            self.recall_with_stats(
                context, shared_knowledge_fresh=shared_knowledge_fresh
            ).candidates
        )

    def recall_with_stats(
        self, context: RecallContext, *, shared_knowledge_fresh: bool = False
    ) -> RecallRun:
        with self._database_lock:
            try:
                if not shared_knowledge_fresh:
                    return RecallRun.empty()
                freshness = self.projection_freshness(shared_knowledge_fresh=True)
                if not freshness.ready:
                    self._rebuild_locked()
                with self._connection() as connection:
                    return recall_with_stats(connection, context)
            except (DatabaseCorruptionError, sqlite3.DatabaseError):
                try:
                    self._rebuild_locked()
                    with self._connection() as connection:
                        return recall_with_stats(connection, context)
                except (DatabaseCorruptionError, OSError, sqlite3.DatabaseError):
                    return RecallRun.empty()
