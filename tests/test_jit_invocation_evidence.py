from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

from shared_memory_runtime import (
    JITInvocationEvidence,
    PolicyProvider,
    TaskPolicyContext,
    audit_jit_invocation_evidence,
    jit_invocation_evidence_path,
    persist_jit_invocation_evidence,
    prepare_formal_task_jit,
    read_jit_invocation_evidence,
)


TASK_ID = "71b6b889-e4f3-4227-9437-661a1630ae59"
SKILLS = ("backend-development", "skill-sync")


def _context(**providers) -> TaskPolicyContext:
    return TaskPolicyContext(
        task_intent="Implement a deterministic feature",
        available_skills=SKILLS,
        existing_providers={
            path: (PolicyProvider("existing_default", value, 10),)
            for path, value in providers.items()
        },
    )


def _prepare(root: Path, *, context: Optional[TaskPolicyContext] = None, **kwargs):
    return prepare_formal_task_jit(
        TASK_ID,
        context=context or _context(),
        codex_home=root,
        **kwargs,
    )


def _evidence(root: Path, **kwargs) -> JITInvocationEvidence:
    result = _prepare(root, **kwargs)
    assert result.invocation_evidence is not None
    return result.invocation_evidence


def _run_writer(root: Path, input_path: Path) -> subprocess.Popen[str]:
    code = """
import json
import sys
from pathlib import Path
from shared_memory_runtime import JITInvocationEvidence, persist_jit_invocation_evidence
payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
evidence = JITInvocationEvidence.from_mapping(payload)
outcome = persist_jit_invocation_evidence(evidence, codex_home=Path(sys.argv[2]))
print(outcome.status)
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return subprocess.Popen(
        [sys.executable, "-c", code, str(input_path), str(root)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _run_two_writers(root: Path, first: JITInvocationEvidence, second: JITInvocationEvidence):
    input_one = root / "writer-one.json"
    input_two = root / "writer-two.json"
    input_one.write_text(json.dumps(first.as_mapping()), encoding="utf-8")
    input_two.write_text(json.dumps(second.as_mapping()), encoding="utf-8")
    processes = [_run_writer(root, input_one), _run_writer(root, input_two)]
    results = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr
        results.append(stdout.strip())
    return results


def test_first_write_is_created_and_contains_only_hashes(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    path = jit_invocation_evidence_path(TASK_ID, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert persist_jit_invocation_evidence(evidence, codex_home=tmp_path).status == "reused"
    assert payload["base_policy_hash"]
    assert payload["adapted_policy_hash"]
    assert payload["activation_result_hash"]
    assert set(payload) == set(evidence.as_mapping())
    assert "policy" not in payload
    assert "active_policy" not in payload
    assert "experience_body" not in payload
    assert "prompt" not in payload


def test_observational_fields_are_excluded_and_first_values_are_preserved(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    candidate = replace(
        evidence,
        prepared_at="2020-01-01T00:00:00Z",
        policy_compile_ms=(evidence.policy_compile_ms or 0.0) + 500.0,
    )
    outcome = persist_jit_invocation_evidence(candidate, codex_home=tmp_path)
    assert outcome.status == "reused"
    stored = read_jit_invocation_evidence(TASK_ID, codex_home=tmp_path)
    assert stored is not None
    assert stored.prepared_at == evidence.prepared_at
    assert stored.policy_compile_ms == evidence.policy_compile_ms


def test_unordered_policy_field_order_is_canonicalized(tmp_path: Path) -> None:
    evidence = _evidence(
        tmp_path,
        evidence_metrics={
            "policy_applied_fields": ["validation.full_suite", "memory.scope"],
        },
    )
    candidate = replace(
        evidence,
        policy_applied_fields=tuple(reversed(evidence.policy_applied_fields or ())),
        runtime_enforced_fields=(
            tuple(reversed(evidence.runtime_enforced_fields))
            if evidence.runtime_enforced_fields is not None
            else None
        ),
        protocol_consumed_fields=tuple(reversed(evidence.protocol_consumed_fields or ())),
    )
    assert persist_jit_invocation_evidence(candidate, codex_home=tmp_path).status == "reused"


def test_policy_value_change_changes_hash_and_conflicts(tmp_path: Path) -> None:
    first = _evidence(tmp_path, context=_context(**{"exploration.max_read_lines": 200}))
    other = tmp_path / "other"
    other.mkdir()
    second = _evidence(other, context=_context(**{"exploration.max_read_lines": 400}))
    assert first.base_policy_hash != second.base_policy_hash
    assert first.adapted_policy_hash != second.adapted_policy_hash
    before = jit_invocation_evidence_path(TASK_ID, tmp_path).read_bytes()
    outcome = persist_jit_invocation_evidence(second, codex_home=tmp_path)
    assert outcome.status == "conflict"
    assert jit_invocation_evidence_path(TASK_ID, tmp_path).read_bytes() == before


def test_advice_value_change_conflicts_even_with_same_adapted_count(tmp_path: Path) -> None:
    candidate = {
        "experience_id": "experience-1",
        "canonical_id": "experience-1",
        "capsule": "bounded capsule",
        "score": 1.0,
        "effective_experience_verification": "verified",
        "remote_verified": True,
        "exact_anchor_hits": 1,
        "applicability_passed": True,
    }
    from shared_memory_runtime import RecallCandidate

    recall = RecallCandidate(**candidate)
    first = _evidence(
        tmp_path,
        ranked_candidates=(recall,),
        advice_by_experience={"experience-1": {"validation": {"full_suite": True}}},
    )
    other = tmp_path / "other"
    other.mkdir()
    second = _evidence(
        other,
        ranked_candidates=(recall,),
        advice_by_experience={"experience-1": {"validation": {"focused_tests": False}}},
    )
    assert first.adapted_field_count == second.adapted_field_count == 1
    assert first.adapted_policy_hash != second.adapted_policy_hash
    assert persist_jit_invocation_evidence(second, codex_home=tmp_path).status == "conflict"


def test_activation_semantic_change_conflicts_with_same_field_names(tmp_path: Path) -> None:
    first = _evidence(tmp_path, user_requirements={"memory": {"max_capsules": 2}})
    other = tmp_path / "other"
    other.mkdir()
    second = _evidence(other, user_requirements={"memory": {"max_capsules": 3}})
    assert first.policy_applied_fields == second.policy_applied_fields
    assert first.activation_result_hash != second.activation_result_hash
    assert persist_jit_invocation_evidence(second, codex_home=tmp_path).status == "conflict"


def test_concurrent_same_content_converges_without_overwrite(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path / "source")
    results = _run_two_writers(tmp_path, evidence, evidence)
    assert sorted(results) == ["created", "reused"]
    assert len(list((tmp_path / ".state/experience-runtime/jit-invocation").glob("*.json"))) == 1


def test_concurrent_different_content_has_one_winner(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path / "source")
    other = replace(evidence, policy_task_class="bug_fix")
    results = _run_two_writers(tmp_path, evidence, other)
    assert sorted(results) == ["conflict", "created"]
    stored = read_jit_invocation_evidence(TASK_ID, codex_home=tmp_path)
    assert stored is not None
    assert stored.policy_task_class in {evidence.policy_task_class, other.policy_task_class}


def test_corrupt_existing_artifact_is_preserved(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    path = jit_invocation_evidence_path(TASK_ID, tmp_path)
    path.write_text("{not-json", encoding="utf-8")
    before = path.read_bytes()
    outcome = persist_jit_invocation_evidence(evidence, codex_home=tmp_path)
    assert outcome.status == "conflict"
    assert outcome.reason == "invalid_existing_evidence"
    assert path.read_bytes() == before


def test_audit_detects_conflicting_current_result_and_accepts_matching_metrics(tmp_path: Path) -> None:
    result = _prepare(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    conflicting = _prepare(
        other,
        context=_context(**{"exploration.max_read_lines": 400}),
    )
    conflict = audit_jit_invocation_evidence(
        TASK_ID,
        codex_home=tmp_path,
        current_result=conflicting,
    )
    assert conflict.status == "JIT Evidence Conflict"

    stored = read_jit_invocation_evidence(TASK_ID, codex_home=tmp_path)
    assert stored is not None
    matching = audit_jit_invocation_evidence(
        TASK_ID,
        codex_home=tmp_path,
        terminal_metrics={
            key: value
            for key, value in stored.as_mapping().items()
            if key not in {"prepared_at", "policy_compile_ms"}
        },
    )
    assert matching.status == "JIT Pipeline Executed"
    assert matching.source == "durable_artifact"


def test_audit_marks_field_only_metrics_insufficient(tmp_path: Path) -> None:
    audit = audit_jit_invocation_evidence(
        TASK_ID,
        codex_home=tmp_path,
        terminal_metrics={
            "task_id": TASK_ID,
            "jit_policy_evidence_version": "jit-invocation-evidence-v1",
            "policy_task_class": "maintenance",
        },
    )
    assert audit.status == "Unable To Prove"
    assert audit.insufficient_sources == ("terminal_metrics",)


def test_invocation_evidence_survives_destroyed_execution_context(tmp_path: Path) -> None:
    _prepare(tmp_path)
    code = """
import sys
from pathlib import Path
from shared_memory_runtime import audit_jit_invocation_evidence
result = audit_jit_invocation_evidence(sys.argv[1], codex_home=Path(sys.argv[2]))
print(result.status)
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, TASK_ID, str(tmp_path)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "JIT Pipeline Executed"


def test_non_formal_interaction_does_not_create_artifact(tmp_path: Path) -> None:
    from shared_memory_runtime import build_jit_policy_evidence

    evidence = build_jit_policy_evidence(formal_task=False)
    assert evidence["formal_task"] is False
    assert not (tmp_path / ".state/experience-runtime/jit-invocation").exists()


def test_no_replace_publish_implementation_has_no_overwrite_primitive() -> None:
    source = Path(__file__).resolve().parents[1] / "src/shared_memory_runtime/jit_invocation_evidence.py"
    assert "os.replace" not in source.read_text(encoding="utf-8")
