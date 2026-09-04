from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from shared_memory_runtime.finalization_receipt import verify_finalization_receipt
from shared_memory_runtime.finalization_reconciliation import (
    EVIDENCE_CONFLICT,
    RECOVERABLE_OUTSTANDING,
    STALE_LOCAL_STATE,
    TERMINAL,
    UNRESOLVED_LOCAL_TASK,
    reconcile_finalization_states,
)
from shared_memory_runtime.markdown import project_key


PROJECT = "github.com/example/project"
PROJECT_KEY = project_key(PROJECT)
TASK_ID = "00000000-0000-4000-8000-000000000001"
FRAGMENT_ID = "01K00000000000000000000001"

FRAGMENT = f"""---
id: {FRAGMENT_ID}
type: task
scope: project
project: {PROJECT}
task_id: {TASK_ID}
status: completed
confidence: high
importance: 4
title: Reconciliation fixture
summary: Verify cross-device finalization evidence.
experience:
  outcome: NEW
  verification: verified
  canonical_id: {FRAGMENT_ID}
  capsule: Use the authoritative shared evidence fixture.
  applies_when:
    project_keys: []
    task_kinds: []
    required_anchors: []
  does_not_apply_when:
    project_keys: []
    task_kinds: []
    excluded_anchors: []
created_at: 2026-09-04T00:00:00Z
---

## Result

Passed with durable evidence.

## Validation

The shared evidence checks passed.

## Evidence

The authoritative remote contains this source.

## Experience

### Trigger

Cross-device stale local State.

### Worked

Reuse shared-side evidence without claiming a local Receipt.

### Does not apply when

None.

### Exceptions

None.
"""


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def make_repo(tmp_path: Path, with_primary: bool = True) -> tuple[Path, Path, Path | None]:
    source = tmp_path / "skills"
    remote = tmp_path / "remote.git"
    source.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(source)], check=True)
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    run_git(source, "config", "user.email", "reconcile@test.invalid")
    run_git(source, "config", "user.name", "Reconcile Test")
    run_git(source, "remote", "add", "origin", str(remote))
    (source / "README.md").write_text("reconciliation fixture\n", encoding="utf-8")
    primary = None
    if with_primary:
        primary = source / ".memory" / "projects" / PROJECT_KEY / "tasks" / "primary.md"
        primary.parent.mkdir(parents=True)
        primary.write_text(FRAGMENT, encoding="utf-8")
    run_git(source, "add", "--", ".")
    run_git(source, "commit", "-m", "fixture")
    run_git(source, "push", "-u", "origin", "main")
    return source, tmp_path / "states", primary


def write_state(
    state_root: Path,
    source: Path,
    *,
    finalization_state: str = "verified_pending",
    learning_stage: str = "sync_completed",
    task_result: str = "Passed",
    primary: str = "Found",
    primary_hash: str = "Unknown",
) -> Path:
    state_root.mkdir(parents=True, exist_ok=True)
    worktree_root = state_root.parent / "worktree"
    worktree_key = hashlib.sha256(
        ("test\0" + str(worktree_root.resolve())).encode("utf-8")
    ).hexdigest()[:12]
    commit = run_git(source, "rev-parse", "HEAD")
    summary = (
        "Recovery Evidence:\n"
        f"Task Result: {task_result}\n"
        f"Primary: {primary}\n"
        f"Primary Hash: {primary_hash}\n"
        "Missing Requirements: None\n"
        f"Git Evidence: Commit={commit}; Push=Success; Remote Verify=Success; Reachable=Yes\n"
    )
    path = state_root / f"{PROJECT_KEY}--{worktree_key}--{TASK_ID}.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "task_id": TASK_ID,
                "project_identity": PROJECT,
                "project_identity_scope": "cross_device",
                "project_key": PROJECT_KEY,
                "project_environment": "test",
                "project_root": str(worktree_root),
                "worktree_root": str(worktree_root),
                "worktree_key": worktree_key,
                "goal": "Cross-device reconciliation fixture",
                "strong_anchors": ["Recovery Inbox"],
                "main_skill": "skill-sync",
                "start_revision": commit,
                "start_worktree_state": {
                    "availability": "available",
                    "kind": "git",
                    "head": commit,
                    "status_digest": "0" * 64,
                    "staged": [],
                    "unstaged": [],
                    "untracked": [],
                },
                "current_revision": commit,
                "changed_area": [],
                "validation_summary": summary,
                "finalization_state": finalization_state,
                "learning_stage": learning_stage,
                "blocked_reason": None,
                "created_at": "2026-09-04T00:00:00Z",
                "updated_at": "2026-09-04T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_shared_complete_is_stale_not_terminal_or_outstanding(tmp_path: Path) -> None:
    source, state_root, primary = make_repo(tmp_path)
    assert primary is not None
    state = write_state(
        state_root,
        source,
        primary_hash=hashlib.sha256(primary.read_bytes()).hexdigest(),
    )
    before = state.read_bytes()

    report = reconcile_finalization_states(source, state_root)
    entry = report.entries[0]

    assert entry.classification == STALE_LOCAL_STATE
    assert entry.action == "CONTINUE"
    assert entry.shared_evidence_complete is True
    assert entry.local_terminal_receipt == "not_proven"
    assert report.stale_local_states == (TASK_ID,)
    assert report.recoverable_outstanding == ()
    assert "terminal" not in entry.as_mapping()
    assert "complete" not in entry.as_mapping()
    assert state.read_bytes() == before

    receipt = verify_finalization_receipt(source, state)
    assert receipt.complete is False
    assert "terminal_state_not_complete" in receipt.reason_codes
    assert set(receipt.as_mapping()) == {
        "task_id",
        "task_result",
        "experience_outcome",
        "primary_path",
        "primary_id",
        "shared_commit",
        "push_verified",
        "remote_verified",
        "finalization_state",
        "learning_stage",
        "complete",
        "reason_codes",
    }


def test_missing_remote_primary_is_recoverable_outstanding(tmp_path: Path) -> None:
    source, state_root, _ = make_repo(tmp_path, with_primary=False)
    state = write_state(state_root, source, primary="Missing")

    report = reconcile_finalization_states(source, state_root)

    assert report.entries[0].classification == RECOVERABLE_OUTSTANDING
    assert report.entries[0].action == "RESUME"
    assert report.recoverable_outstanding == (TASK_ID,)


def test_primary_hash_conflict_is_blocked(tmp_path: Path) -> None:
    source, state_root, primary = make_repo(tmp_path)
    assert primary is not None
    state = write_state(state_root, source, primary_hash="0" * 64)

    report = reconcile_finalization_states(source, state_root)

    assert report.entries[0].classification == EVIDENCE_CONFLICT
    assert report.entries[0].action == "BLOCK"
    assert "primary_hash_conflict" in report.entries[0].reason_codes


def test_remote_blob_hash_conflict_is_blocked(tmp_path: Path) -> None:
    source, state_root, primary = make_repo(tmp_path)
    assert primary is not None
    primary.write_text(FRAGMENT.replace("Passed with durable evidence.", "Changed locally."), encoding="utf-8")
    state = write_state(
        state_root,
        source,
        primary_hash=hashlib.sha256(primary.read_bytes()).hexdigest(),
    )

    report = reconcile_finalization_states(source, state_root)

    assert report.entries[0].classification == EVIDENCE_CONFLICT
    assert "remote_primary_hash_conflict" in report.entries[0].reason_codes


def test_undetermined_pending_state_is_preserved(tmp_path: Path) -> None:
    source, state_root, _ = make_repo(tmp_path, with_primary=False)
    state = write_state(
        state_root,
        source,
        finalization_state="pending",
        learning_stage="implementation",
        task_result="Undetermined",
        primary="Missing",
    )

    report = reconcile_finalization_states(source, state_root)

    assert report.entries[0].classification == UNRESOLVED_LOCAL_TASK
    assert report.entries[0].action == "PRESERVE"


def test_local_terminal_requires_authoritative_receipt(tmp_path: Path) -> None:
    source, state_root, primary = make_repo(tmp_path)
    assert primary is not None
    state = write_state(
        state_root,
        source,
        finalization_state="sedimented",
        learning_stage="remote_verified",
        primary_hash=hashlib.sha256(primary.read_bytes()).hexdigest(),
    )

    report = reconcile_finalization_states(source, state_root)

    assert report.entries[0].classification == TERMINAL
    assert report.entries[0].local_terminal_receipt == "proven"
    assert report.entries[0].shared_evidence_complete is True


def test_missing_explicit_candidate_is_preserved_without_fabrication(tmp_path: Path) -> None:
    source, state_root, _ = make_repo(tmp_path, with_primary=False)
    state_root.mkdir(parents=True, exist_ok=True)
    before = sorted(state_root.iterdir())

    report = reconcile_finalization_states(
        source,
        state_root,
        task_ids=["d41b16ba-28fc-4e3f-a5ef-18ffd99d94df"],
    )

    assert report.entries[0].task_id == "d41b16ba-28fc-4e3f-a5ef-18ffd99d94df"
    assert report.entries[0].classification == UNRESOLVED_LOCAL_TASK
    assert report.entries[0].action == "PRESERVE"
    assert sorted(state_root.iterdir()) == before
