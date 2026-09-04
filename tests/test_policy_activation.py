from __future__ import annotations

from shared_memory_runtime import (
    TaskPolicyContext,
    activate_task_policy,
    compile_task_policy,
)


def _policy() -> dict:
    result = compile_task_policy(
        TaskPolicyContext(
            task_intent="Implement a feature",
            available_skills=("backend-development",),
        )
    )
    assert result.policy is not None
    return result.policy


def test_j4_activation_is_protocol_only_and_separates_evidence() -> None:
    result = activate_task_policy(
        _policy(),
        available_skills=("backend-development",),
        actual_metrics={
            "shell_calls": 3,
            "file_reads": 5,
            "tool_output_bytes": 1200,
        },
    )
    assert result.fallback is False
    assert result.policy_activation_enforcement == "Protocol Only"
    assert result.host_enforced is False
    assert result.technically_unbypassable is False
    assert result.observed_actual_behavior == {
        "shell_calls": 3,
        "file_reads": 5,
        "tool_output_bytes": 1200,
    }
    assert "planning.depth" in result.not_instrumented_fields
    assert "capabilities.skills" in result.not_instrumented_fields
    assert "execution.shell_batching" in result.not_instrumented_fields
    assert result.runtime_enforced_capabilities == ()


def test_j4_user_requirement_overrides_policy_but_not_kernel() -> None:
    result = activate_task_policy(
        _policy(),
        available_skills=("backend-development",),
        project_requirements={"validation": {"full_suite": False}},
        user_requirements={"validation": {"full_suite": True}},
    )
    assert result.fallback is False
    assert result.policy_source == "user_requirement"
    assert result.active_policy["validation"]["full_suite"] is True

    blocked = activate_task_policy(
        _policy(),
        available_skills=("backend-development",),
        user_requirements={"action_gate": False},
    )
    assert blocked.fallback is True
    assert blocked.active_policy is None
    assert "kernel_advice_rejected" in blocked.reason_codes


def test_j4_invalid_policy_falls_back_to_existing_behavior() -> None:
    policy = _policy()
    policy["planning"]["depth"] = "not-authoritative"
    result = activate_task_policy(policy, available_skills=("backend-development",))
    assert result.fallback is True
    assert result.active_policy is None
    assert result.policy_source == "existing_default"
    assert "policy_invalid" in result.reason_codes


def test_j4_uninstrumented_agent_claim_is_not_observed_evidence() -> None:
    result = activate_task_policy(
        _policy(),
        available_skills=("backend-development",),
        actual_metrics={"agent_execution_strategy": "parallel"},
    )
    assert result.observed_actual_behavior == {}
    assert "unrecognized_actual_metric" in result.reason_codes
