from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Optional

from shared_memory_runtime import (
    Applicability,
    DraftExperience,
    ExperienceProjector,
    RecallContext,
    TaskMetrics,
    compile_terminal_experience,
    record_task_metrics,
)
from shared_memory_runtime.markdown import project_key


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def make_id(number: int) -> str:
    return "01K" + "0" * 19 + f"{number:04X}"


def fragment(
    fragment_id: str,
    *,
    fragment_type: str = "task",
    scope: str = "project",
    project: Optional[str] = "github.com/example/project",
    status: str = "completed",
    verification: str = "verified",
    outcome: str = "NEW",
    canonical_id: Optional[str] = None,
    title: str = "Index lock ownership",
    summary: str = "Classify lock ownership before changing the file.",
    capsule: str = "When an index lock anchor appears, classify ownership from evidence before acting and preserve exceptions.",
    trigger: str = "A git index lock blocks a formal task.",
    core_action: str = "Inspect ownership evidence before removing or changing the lock.",
    exceptions: str = "Never remove an actively owned lock.",
    does_not_apply: str = "None",
    anchors: tuple[str, ...] = ("git/index.lock",),
    applies_when: Optional[Applicability] = None,
    feedback: Optional[tuple[bool, str]] = None,
    supersedes: tuple[str, ...] = (),
    task_id: Optional[str] = None,
) -> str:
    canonical_id = canonical_id or fragment_id
    task_id = task_id if task_id is not None else ("task-" + fragment_id[-8:] if fragment_type in {"task", "case", "handoff"} else None)
    applicability = applies_when or Applicability()
    project_line = f"project: {project}\n" if scope == "project" else ""
    task_line = f"task_id: {task_id}\n" if task_id else ""
    feedback_block = ""
    if feedback is not None:
        feedback_block = f"  feedback:\n    reused: {'true' if feedback[0] else 'false'}\n    result: {feedback[1]}\n"
    supersedes_yaml = "[]" if not supersedes else "\n" + "\n".join(f"    - {value}" for value in supersedes)
    applies_project = ", ".join(applicability.project_keys)
    applies_kind = ", ".join(applicability.task_kinds)
    required_anchor = ", ".join(applicability.required_anchors)
    excluded_project = ", ".join(applicability.excluded_project_keys)
    excluded_kind = ", ".join(applicability.excluded_task_kinds)
    excluded_anchor = ", ".join(applicability.excluded_anchors)
    return f"""---
id: {fragment_id}
type: {fragment_type}
scope: {scope}
{project_line}{task_line}status: {status}
confidence: high
importance: 4
title: {title}
summary: {summary}
anchors:
  files:
    - {anchors[0]}
related:
  supersedes: {supersedes_yaml}
source:
  paths: []
  revisions: []
experience:
  outcome: {outcome}
  verification: {verification}
  canonical_id: {canonical_id}
  capsule: "{capsule}"
  applies_when:
    project_keys: [{applies_project}]
    task_kinds: [{applies_kind}]
    required_anchors: [{required_anchor}]
  does_not_apply_when:
    project_keys: [{excluded_project}]
    task_kinds: [{excluded_kind}]
    excluded_anchors: [{excluded_anchor}]
{feedback_block}created_at: 2026-08-31T00:00:00Z
---

# Goal

Validate this Experience path.

## Result

Passed the formal task result with terminal evidence.

## Validation summary

Validation passed against the authoritative source.

## Experience

### Trigger

{trigger}

### Worked

{core_action}

### Does not apply when

{does_not_apply}

### Exceptions

{exceptions}

### Evidence

Terminal evidence and validation support the recorded Experience.
"""


def source_repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, Path]:
    source = tmp_path / "skills"
    remote = tmp_path / "remote.git"
    source.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(source)], check=True)
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    run_git(source, "config", "user.email", "runtime@test.invalid")
    run_git(source, "config", "user.name", "Runtime Test")
    run_git(source, "remote", "add", "origin", str(remote))
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    run_git(source, "add", "--", ".")
    run_git(source, "commit", "-m", "fixture")
    run_git(source, "push", "-u", "origin", "main")
    return source, remote


def test_status_and_remote_gate(tmp_path: Path) -> None:
    source, _ = source_repo(
        tmp_path,
        {
            ".memory/projects/example/tasks/candidate.md": fragment(
                make_id(1), status="completed", verification="candidate"
            ),
            ".memory/projects/example/tasks/verified.md": fragment(make_id(2)),
        },
    )
    database = tmp_path / "experience.db"
    projector = ExperienceProjector(source, database)
    report = projector.rebuild()
    assert len(report.projected) == 2
    assert all(proof.verified for proof in report.proofs)
    context = RecallContext(
        project_key=project_key("github.com/example/project"),
        anchors=("git/index.lock",),
        query="index lock",
    )
    candidates = projector.recall(context)
    assert [candidate.experience_id for candidate in candidates] == [make_id(2)]

    changed = source / ".memory/projects/example/tasks/verified.md"
    changed.write_text(changed.read_text(encoding="utf-8").replace("verification: verified", "verification: candidate"), encoding="utf-8")
    failed = ExperienceProjector(source, tmp_path / "failed.db").rebuild()
    failed_row = next(record for record in failed.projected if record.id == make_id(2))
    failed_proof = next(proof for proof in failed.proofs if proof.source_path == failed_row.source_path)
    assert not failed_proof.verified
    assert failed_proof.reason_code == "containing_revision_unavailable"
    assert ExperienceProjector(source, tmp_path / "failed.db").recall(context) == []


def test_missing_remote_is_not_remote_verified(tmp_path: Path) -> None:
    source, _ = source_repo(
        tmp_path,
        {".memory/projects/example/tasks/experience.md": fragment(make_id(3))},
    )
    run_git(source, "remote", "remove", "origin")
    report = ExperienceProjector(source, tmp_path / "experience.db").rebuild()
    assert len(report.projected) == 1
    assert report.proofs[0].verified is False
    assert report.proofs[0].reason_code == "upstream_unavailable"


def test_projection_schema_uses_explicit_status_fields(tmp_path: Path) -> None:
    source, _ = source_repo(
        tmp_path,
        {".memory/projects/example/tasks/experience.md": fragment(make_id(4))},
    )
    database = tmp_path / "experience.db"
    ExperienceProjector(source, database).rebuild()
    import sqlite3

    connection = sqlite3.connect(database)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(experiences)")}
    assert "fragment_status" in columns
    assert "experience_verification" in columns
    assert "status" not in columns
    assert "verification" not in columns
    assert {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")} >= {
        "experiences",
        "experience_anchors",
        "experience_feedback",
        "experience_fts",
    }
    connection.close()


def test_global_and_project_scope_are_filtered_before_ranking(tmp_path: Path) -> None:
    assert project_key("github.com/rovdex/skills") == "rovdex-skills-94aff2f9"
    one = fragment(make_id(10), project="github.com/example/one", title="One project", summary="One project anchor")
    two = fragment(make_id(11), project="github.com/example/two", title="Two project", summary="Two project anchor")
    global_record = fragment(
        make_id(12),
        fragment_type="fact",
        scope="global",
        project=None,
        title="Global lock rule",
        summary="Global lock rule",
    )
    source, _ = source_repo(
        tmp_path,
        {
            ".memory/projects/one/tasks/one.md": one,
            ".memory/projects/two/tasks/two.md": two,
            ".memory/global/patterns/global.md": global_record,
        },
    )
    projector = ExperienceProjector(source, tmp_path / "experience.db")
    projector.rebuild()
    context = RecallContext(
        project_key=project_key("github.com/example/one"),
        anchors=("git/index.lock",),
        query="project",
    )
    ids = {candidate.experience_id for candidate in projector.recall(context)}
    assert make_id(10) in ids
    assert make_id(11) not in ids
    assert make_id(12) in ids


def test_feedback_rebuild_survives_database_deletion(tmp_path: Path) -> None:
    canonical = make_id(20)
    reinforce = make_id(21)
    source, _ = source_repo(
        tmp_path,
        {
            ".memory/projects/example/tasks/canonical.md": fragment(canonical),
            ".memory/projects/example/tasks/reinforce.md": fragment(
                reinforce,
                outcome="REINFORCE",
                canonical_id=canonical,
                feedback=(True, "success"),
                title="Reused index lock ownership",
            ),
        },
    )
    database = tmp_path / "experience.db"
    projector = ExperienceProjector(source, database)
    projector.rebuild()
    import sqlite3

    connection = sqlite3.connect(database)
    assert connection.execute(
        "SELECT reuse_count, success_count, failure_count, last_result FROM experience_feedback WHERE experience_id = ?",
        (canonical,),
    ).fetchone() == (1, 1, 0, "success")
    connection.close()
    database.unlink()
    projector.rebuild()
    connection = sqlite3.connect(database)
    assert connection.execute(
        "SELECT reuse_count, success_count, failure_count, last_result FROM experience_feedback WHERE experience_id = ?",
        (canonical,),
    ).fetchone() == (1, 1, 0, "success")
    connection.close()


def test_correction_supersedes_old_experience_without_rewriting_it(tmp_path: Path) -> None:
    old_id = make_id(30)
    correction_id = make_id(31)
    source, _ = source_repo(
        tmp_path,
        {
            ".memory/projects/example/tasks/old.md": fragment(old_id, core_action="Remove stale lock.", exceptions="None"),
            ".memory/projects/example/tasks/correction.md": fragment(
                correction_id,
                outcome="CORRECT",
                canonical_id=correction_id,
                core_action="Classify ownership and only remove a stale lock.",
                exceptions="Never remove an actively owned lock.",
                supersedes=(old_id,),
                title="Correct lock handling",
                summary="Correction adds ownership classification.",
            ),
        },
    )
    projector = ExperienceProjector(source, tmp_path / "experience.db")
    projector.rebuild()
    import sqlite3

    connection = sqlite3.connect(tmp_path / "experience.db")
    assert connection.execute(
        "SELECT experience_verification FROM experiences WHERE id = ?", (old_id,)
    ).fetchone() == ("verified",)
    connection.close()
    candidates = projector.recall(
        RecallContext(
            project_key=project_key("github.com/example/project"),
            anchors=("git/index.lock",),
            query="lock",
        )
    )
    ids = {candidate.experience_id for candidate in candidates}
    assert correction_id in ids
    assert old_id not in ids


def test_anchor_match_is_candidate_only_and_compiler_is_deterministic() -> None:
    common = Applicability()
    candidate = DraftExperience(
        experience_id=make_id(40),
        scope="project",
        project_key="project-1",
        kind="task",
        trigger="A lock blocks the task.",
        core_action="Classify ownership.",
        applicability=common,
        does_not_apply="None",
        exceptions="Never remove an active lock.",
        anchors=("git/index.lock",),
        reuse_proven=True,
    )
    different = DraftExperience(
        **{
            **candidate.__dict__,
            "experience_id": make_id(41),
            "trigger": "Resolve the path of a worktree lock.",
            "core_action": "Resolve the worktree path.",
        }
    )
    from shared_memory_runtime.compiler import SemanticCandidate

    semantic = SemanticCandidate(
        experience_id=make_id(42),
        canonical_id=make_id(42),
        scope="project",
        project_key="project-1",
        kind="task",
        trigger="A lock blocks the task.",
        core_action="Classify ownership.",
        applicability=common,
        does_not_apply="None",
        exceptions="Never remove an active lock.",
        anchors=("git/index.lock",),
    )
    assert compile_terminal_experience(candidate, [semantic]).outcome == "REINFORCE"
    assert compile_terminal_experience(different, [semantic]).outcome == "NEW"
    correction = DraftExperience(
        **{**candidate.__dict__, "experience_id": make_id(43), "correction_of": make_id(42)}
    )
    result = compile_terminal_experience(correction, [semantic])
    assert result.outcome == "CORRECT"
    assert result.candidate_ids == (make_id(42),)


def test_short_text_fallback_and_capsule_limits(tmp_path: Path) -> None:
    files = {}
    for number in range(50, 58):
        files[f".memory/projects/example/tasks/{number}.md"] = fragment(
            make_id(number),
            title=f"锁文件经验 {number}",
            summary="中文短查询验证锁文件回退。",
            capsule=f"Capsule {number} for the lock experience.",
        )
    source, _ = source_repo(tmp_path, files)
    projector = ExperienceProjector(source, tmp_path / "experience.db")
    projector.rebuild()
    base = dict(project_key=project_key("github.com/example/project"), anchors=("git/index.lock",), query="锁文")
    assert len(projector.recall(RecallContext(**base))) == 4
    assert len(projector.recall(RecallContext(**base, task_class="complex"))) == 5
    assert len(projector.recall(RecallContext(**base, task_class="debugging", expand=True))) == 6


def test_corruption_quarantine_and_atomic_rebuild(tmp_path: Path) -> None:
    source, _ = source_repo(
        tmp_path,
        {".memory/projects/example/tasks/experience.md": fragment(make_id(60))},
    )
    database = tmp_path / "experience.db"
    projector = ExperienceProjector(source, database)
    projector.rebuild()
    database.write_bytes(b"not a sqlite database")
    Path(str(database) + "-wal").write_bytes(b"old wal")
    Path(str(database) + "-shm").write_bytes(b"old shm")

    recovered = projector.rebuild()

    assert recovered.corruption_recovered is True
    assert len(recovered.quarantine_paths) == 3
    assert database.exists()
    connection = sqlite3.connect(database)
    assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    connection.close()
    assert projector.recall(
        RecallContext(
            project_key=project_key("github.com/example/project"),
            anchors=("git/index.lock",),
            query="index lock",
        )
    )

    for _ in range(3):
        database.write_bytes(b"corrupt again")
        projector.rebuild()
    quarantine_databases = sorted(tmp_path.glob("experience.db.corrupt-*"))
    assert len(quarantine_databases) == 3


def test_projection_failure_preserves_authoritative_source(tmp_path: Path) -> None:
    source, _ = source_repo(
        tmp_path,
        {".memory/projects/example/tasks/experience.md": fragment(make_id(61))},
    )
    database = tmp_path / "experience.db"
    projector = ExperienceProjector(source, database)
    source_file = source / ".memory/projects/example/tasks/experience.md"
    before = source_file.read_bytes()

    import shared_memory_runtime.projector as projector_module

    original_open_database = projector_module.open_database

    def fail_open_database(_database_path: Path):
        raise sqlite3.OperationalError("injected projection failure")

    projector_module.open_database = fail_open_database
    try:
        try:
            projector.project_path(".memory/projects/example/tasks/experience.md")
        except sqlite3.OperationalError:
            pass
        else:
            raise AssertionError("projection failure must remain visible")
    finally:
        projector_module.open_database = original_open_database

    assert source_file.read_bytes() == before
    assert run_git(source, "status", "--porcelain") == ""


def test_recall_stats_and_local_only_metrics(tmp_path: Path) -> None:
    source, _ = source_repo(
        tmp_path,
        {".memory/projects/example/tasks/experience.md": fragment(make_id(62))},
    )
    database = tmp_path / "experience.db"
    projector = ExperienceProjector(source, database)
    projector.rebuild()
    run = projector.recall_with_stats(
        RecallContext(
            project_key=project_key("github.com/example/project"),
            anchors=("git/index.lock",),
            query="index lock",
        )
    )
    assert len(run.candidates) == 1
    assert run.stats.candidate_count >= 1
    assert run.stats.capsule_count == 1
    assert run.stats.approx_memory_tokens > 0
    assert run.stats.full_markdown_expansions == 0

    codex_home = tmp_path / "codex"
    metrics = TaskMetrics(
        task_id="task-62",
        project_key=project_key("github.com/example/project"),
        task_result="Passed",
        started_at="2026-08-31T00:00:00Z",
        finished_at="2026-08-31T00:00:01Z",
        recall_ms=1.25,
        candidate_count=run.stats.candidate_count,
        capsule_count=run.stats.capsule_count,
        approx_memory_tokens=run.stats.approx_memory_tokens,
        extra_learning_calls=0,
        compiler_ms=0.5,
        sqlite_transactions=1,
        finalization_ms=3.0,
    )
    assert record_task_metrics(metrics, codex_home) is True
    metric_file = codex_home / ".state/experience-runtime/metrics.jsonl"
    row = json.loads(metric_file.read_text(encoding="utf-8"))
    assert set(row) == set(metrics.as_mapping())
    assert "capsule" not in row
    invalid_home = tmp_path / "not-a-directory"
    invalid_home.write_text("file", encoding="utf-8")
    assert record_task_metrics(
        TaskMetrics(
            task_id="task-63",
            project_key=None,
            task_result="Passed",
            started_at="2026-08-31T00:00:00Z",
            finished_at="2026-08-31T00:00:01Z",
        ),
        invalid_home,
    ) is False


def test_projection_freshness_is_separate_from_runtime_compatibility(tmp_path: Path) -> None:
    source, _ = source_repo(
        tmp_path,
        {".memory/projects/example/tasks/experience.md": fragment(make_id(63))},
    )
    database = tmp_path / "experience.db"
    projector = ExperienceProjector(source, database)

    assert projector.projection_freshness().reason_code == "projection_missing"
    projector.rebuild()
    assert projector.projection_freshness().ready is True

    source_file = source / ".memory/projects/example/tasks/experience.md"
    source_file.write_text(
        source_file.read_text(encoding="utf-8").replace("Index lock ownership", "Changed lock ownership"),
        encoding="utf-8",
    )
    freshness = projector.projection_freshness()
    assert freshness.ready is False
    assert freshness.reason_code == "projection_source_mismatch"
