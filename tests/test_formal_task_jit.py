from __future__ import annotations

import json

import shared_memory_runtime.formal_task_jit as jit_pipeline
from shared_memory_runtime import (
    PolicyProvider,
    RecallCandidate,
    TaskMetrics,
    TaskPolicyContext,
    prepare_formal_task_jit,
    record_final_jit_policy_metrics,
)


TASK_ID = "62b6b889-e4f3-4227-9437-661a1630ae59"
SKILLS = ("backend-development", "skill-sync")


def _context() -> TaskPolicyContext:
    return TaskPolicyContext(
        task_intent="Implement a deterministic feature",
        available_skills=SKILLS,
    )


def _candidate(experience_id: str, score: float = 1.0) -> RecallCandidate:
    return RecallCandidate(
        experience_id=experience_id,
        canonical_id=experience_id,
        capsule="bounded capsule",
        score=score,
        effective_experience_verification="verified",
        remote_verified=True,
        exact_anchor_hits=1,
        applicability_passed=True,
    )


def test_formal_task_pipeline_order_and_activation_handoff() -> None:
    events = []
    original_compile = jit_pipeline.compile_task_policy
    original_adapt = jit_pipeline.adapt_task_policy
    original_activate = jit_pipeline.activate_task_policy

    def compile_wrapper(context):
        events.append("j2")
        return original_compile(context)

    def adapt_wrapper(*args, **kwargs):
        events.append("j3")
        return original_adapt(*args, **kwargs)

    def activate_wrapper(*args, **kwargs):
        events.append("j4")
        return original_activate(*args, **kwargs)

    jit_pipeline.compile_task_policy = compile_wrapper
    jit_pipeline.adapt_task_policy = adapt_wrapper
    jit_pipeline.activate_task_policy = activate_wrapper
    try:
        result = prepare_formal_task_jit(TASK_ID, context=_context())

        def execute(activation):
            events.append("execution")
            assert activation is result.activation_result
            assert activation is not None
            assert activation.active_policy is not None
            return activation.policy_source

        assert result.consume_for_execution(execute) == "validated_jit_task_policy"
    finally:
        jit_pipeline.compile_task_policy = original_compile
        jit_pipeline.adapt_task_policy = original_adapt
        jit_pipeline.activate_task_policy = original_activate

    assert events == ["j2", "j3", "j4", "execution"]
    assert result.j2_executed is True
    assert result.j3_executed is True
    assert result.j4_executed is True
    assert result.evidence_for_task()["task_id"] == TASK_ID
    assert result.evidence_for_task()["j2"]["task_policy"] == "Compiled"
    assert result.evidence_for_task()["j3"]["experience_adaptation"] == "No Eligible Advice"
    assert result.evidence_for_task()["j4"]["policy_activation"] == "Active"


def test_eligible_advice_and_metrics_are_correlated_once(tmp_path) -> None:
    result = prepare_formal_task_jit(
        TASK_ID,
        context=_context(),
        ranked_candidates=(_candidate("experience-1"),),
        advice_by_experience={"experience-1": {"validation": {"full_suite": True}}},
        evidence_metrics={"experience_advice_count": 1, "experience_conflict_count": 0},
        actual_metrics={"file_reads": 4},
    )
    assert result.evidence_for_task()["j3"]["experience_adaptation"] == "Applied"
    assert result.evidence_for_task()["j3"]["adapted_fields"] == ["validation.full_suite"]
    assert result.evidence_for_task()["j4"]["observed_compliance"] == "Partially Verified"

    metrics = TaskMetrics(
        task_id=TASK_ID,
        project_key=None,
        task_result="Passed",
        started_at="2026-09-05T00:00:00Z",
        finished_at="2026-09-05T00:00:01Z",
        **result.metrics_fields(),
    )
    assert record_final_jit_policy_metrics(metrics, tmp_path) is True
    assert record_final_jit_policy_metrics(metrics, tmp_path) is True
    rows = [
        json.loads(line)
        for line in (tmp_path / ".state/experience-runtime/metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["task_id"] == TASK_ID
    assert rows[0]["experience_advice_count"] == 1
    assert rows[0]["adapted_field_count"] == 1
    assert "policy" not in rows[0]
    assert "capsule" not in rows[0]


def test_j2_failure_keeps_existing_behavior_and_does_not_invent_policy() -> None:
    result = prepare_formal_task_jit(
        TASK_ID,
        context=TaskPolicyContext(
            task_intent="maintenance",
            available_skills=SKILLS,
            existing_providers={
                "memory.max_capsules": (
                    PolicyProvider("existing_default", 1, 1),
                    PolicyProvider("existing_default", 2, 1),
                )
            },
        ),
    )
    assert result.compiler_result.valid is False
    assert result.compiler_result.fallback is True
    assert result.adaptation_result is None
    assert result.activation_result is None
    assert result.fallback_to_existing_behavior is True
    assert result.evidence_for_task()["j2"]["task_policy"] == "Invalid"
    assert result.evidence_for_task()["j3"]["experience_adaptation"] == "Not Run"
    assert result.evidence_for_task()["j4"]["policy_activation"] == "Not Run"
    assert result.consume_for_execution(lambda activation: activation) is None


def test_pipeline_does_not_recompile_per_execution_callback() -> None:
    calls = []
    original_compile = jit_pipeline.compile_task_policy

    def compile_wrapper(context):
        calls.append("compile")
        return original_compile(context)

    jit_pipeline.compile_task_policy = compile_wrapper
    try:
        result = prepare_formal_task_jit(TASK_ID, context=_context())
        result.consume_for_execution(lambda activation: activation)
        result.consume_for_execution(lambda activation: activation)
    finally:
        jit_pipeline.compile_task_policy = original_compile
    assert calls == ["compile"]
