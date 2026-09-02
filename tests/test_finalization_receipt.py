from __future__ import annotations

import json
import subprocess
from pathlib import Path

from shared_memory_runtime.finalization_receipt import verify_finalization_receipt


def _run_git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _fragment(
    fragment_id: str,
    task_id: str,
    *,
    valid_experience: bool = True,
    project: str = "github.com/example/project",
) -> str:
    experience_section = """
## Experience

### Trigger

Terminal formal Task evidence exists.

### Worked

Validate State, Primary, and remote proof before the final response.

### Does not apply when

None.

### Exceptions

None.

### Evidence

State and Git evidence are available.
""" if valid_experience else ""
    return f"""---
id: {fragment_id}
type: task
scope: project
project: {project}
task_id: {task_id}
status: completed
confidence: high
importance: 4
title: Finalization receipt
summary: Verify formal Task terminal evidence from durable sources.
anchors:
  concepts:
    - Finalization Receipt
related:
  supersedes: []
source:
  paths: []
  revisions: []
experience:
  outcome: NEW
  verification: verified
  canonical_id: {fragment_id}
  capsule: "Validate State, exactly one Primary, and remote Git proof before reporting a formal Task terminal."
  applies_when:
    project_keys: []
    task_kinds: []
    required_anchors: []
  does_not_apply_when:
    project_keys: []
    task_kinds: []
    excluded_anchors: []
created_at: 2026-09-02T00:00:00Z
---

# Goal

Verify terminal evidence.

## Result

Passed with durable evidence.

## Validation summary

Validated the terminal State and remote source proof.
{experience_section}
"""


def _source_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    source = tmp_path / "skills"
    remote = tmp_path / "remote.git"
    source.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(source)], check=True)
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    _run_git(source, "config", "user.email", "receipt@test.invalid")
    _run_git(source, "config", "user.name", "Receipt Test")
    _run_git(source, "remote", "add", "origin", str(remote))
    (source / "README.md").write_text("receipt fixture\n", encoding="utf-8")
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _run_git(source, "add", "--", ".")
    _run_git(source, "commit", "-m", "fixture")
    _run_git(source, "push", "-u", "origin", "main")
    return source


def _state(
    path: Path,
    task_id: str,
    *,
    project_key: str = "example-project-3d3378b2",
    finalization_state: str = "sedimented",
    learning_stage: str = "remote_verified",
) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "task_id": task_id,
                "project_key": project_key,
                "finalization_state": finalization_state,
                "learning_stage": learning_stage,
            }
        ),
        encoding="utf-8",
    )


def test_receipt_requires_valid_terminal_state_primary_and_remote_proof(tmp_path: Path) -> None:
    task_id = "task-receipt-complete"
    fragment_id = "01K00000000000000000000001"
    source = _source_repo(
        tmp_path,
        {".memory/projects/example-project-3d3378b2/tasks/primary.md": _fragment(fragment_id, task_id)},
    )
    state = tmp_path / "state.json"
    _state(state, task_id)

    receipt = verify_finalization_receipt(source, state)

    assert receipt.complete is True
    assert receipt.task_result == "Passed"
    assert receipt.primary_id == fragment_id
    assert receipt.experience_outcome == "NEW"
    assert receipt.shared_commit == _run_git(source, "rev-parse", "HEAD")
    assert receipt.push_verified is True
    assert receipt.remote_verified is True
    assert receipt.reason_codes == ("receipt_complete",)
    assert _run_git(source, "status", "--porcelain") == ""


def test_receipt_rejects_missing_primary_and_nonterminal_state(tmp_path: Path) -> None:
    source = _source_repo(tmp_path, {})
    state = tmp_path / "state.json"
    _state(state, "task-missing", finalization_state="verified_pending", learning_stage="sync_completed")

    receipt = verify_finalization_receipt(source, state)

    assert receipt.complete is False
    assert receipt.push_verified is False
    assert receipt.remote_verified is False
    assert set(receipt.reason_codes) == {"terminal_state_not_complete", "primary_missing"}


def test_receipt_rejects_duplicate_primary_invalid_experience_and_unpushed_change(tmp_path: Path) -> None:
    task_id = "task-duplicate"
    source = _source_repo(
        tmp_path,
        {
            ".memory/projects/example-project-3d3378b2/tasks/one.md": _fragment("01K00000000000000000000002", task_id),
            ".memory/projects/example-project-3d3378b2/cases/two.md": _fragment("01K00000000000000000000003", task_id),
        },
    )
    state = tmp_path / "duplicate.json"
    _state(state, task_id)
    duplicate = verify_finalization_receipt(source, state)
    assert duplicate.complete is False
    assert "primary_not_exactly_one" in duplicate.reason_codes

    invalid_task_id = "task-invalid"
    invalid_path = source / ".memory/projects/example-project-3d3378b2/tasks/invalid.md"
    invalid_path.write_text(_fragment("01K00000000000000000000004", invalid_task_id, valid_experience=False), encoding="utf-8")
    _run_git(source, "add", "--", invalid_path.relative_to(source).as_posix())
    _run_git(source, "commit", "-m", "invalid fixture")
    _run_git(source, "push")
    _state(state, invalid_task_id)
    invalid = verify_finalization_receipt(source, state)
    assert invalid.complete is False
    assert "missing_experience_section" in invalid.reason_codes

    unpushed_task_id = "task-unpushed"
    unpushed_path = source / ".memory/projects/example-project-3d3378b2/tasks/unpushed.md"
    unpushed_path.write_text(_fragment("01K00000000000000000000005", unpushed_task_id), encoding="utf-8")
    _run_git(source, "add", "--", unpushed_path.relative_to(source).as_posix())
    _run_git(source, "commit", "-m", "unpushed fixture")
    _state(state, unpushed_task_id)
    unpushed = verify_finalization_receipt(source, state)
    assert unpushed.complete is False
    assert unpushed.push_verified is False
    assert "remote_proof_containing_revision_unavailable" in unpushed.reason_codes


def test_receipt_accepts_historical_non_default_port_project_key(tmp_path: Path) -> None:
    task_id = "task-qdtg-legacy-key"
    fragment_id = "01K00000000000000000000006"
    source = _source_repo(
        tmp_path,
        {
            ".memory/projects/adc-approval-b93c923f/tasks/primary.md": _fragment(
                fragment_id,
                task_id,
                project="qdtg.com:43000/ADC/adc-approval",
            )
        },
    )
    state = tmp_path / "legacy-port.json"
    _state(state, task_id, project_key="adc-approval-b93c923f")

    receipt = verify_finalization_receipt(source, state)

    assert receipt.complete is True
    assert receipt.primary_id == fragment_id
    assert receipt.reason_codes == ("receipt_complete",)
