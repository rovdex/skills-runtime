from __future__ import annotations

from shared_memory_runtime import (
    POLICY_REGISTRY_VERSION,
    PolicyProvider,
    TaskPolicyContext,
    compile_task_policy,
    validate_task_policy,
)


AVAILABLE = (
    "api-development",
    "backend-development",
    "bug-debugging",
    "code-review",
    "development-tooling",
    "prototype-analysis",
    "security-development",
    "skill-development",
    "skill-sync",
)


def test_j2_compiles_without_experience_or_sqlite() -> None:
    result = compile_task_policy(
        TaskPolicyContext(
            task_intent="Implement a deterministic API feature",
            available_skills=AVAILABLE,
        )
    )
    assert result.valid is True
    assert result.fallback is False
    assert result.policy is not None
    assert result.policy["task_class"] == "api"
    assert result.policy["capabilities"]["skills"] == ["api-development", "backend-development"]
    assert result.experience_ignored is False
    assert result.registry_version == POLICY_REGISTRY_VERSION
    assert result.provenance["exploration.max_output_kb"] == "shadow_baseline"


def test_j2_ignores_experience_for_adaptation() -> None:
    context = TaskPolicyContext(
        task_intent="Implement a feature",
        available_skills=("backend-development",),
        experience_metadata={"policy_advice": {"validation": {"full_suite": True}}},
    )
    result = compile_task_policy(context)
    assert result.valid is True
    assert result.experience_ignored is True
    assert result.policy["validation"]["full_suite"] is False
    assert "experience_ignored_for_j2" in result.reason_codes


def test_provider_precedence_and_provenance_are_field_specific() -> None:
    result = compile_task_policy(
        TaskPolicyContext(
            task_intent="Implement a feature",
            available_skills=("backend-development",),
            existing_providers={
                "planning.depth": PolicyProvider("existing_default", "deep", 10),
            },
            skill_providers={
                "planning.depth": PolicyProvider("skill_default", "fast", 10),
                "exploration.max_read_lines": PolicyProvider("skill_default", 160, 1),
            },
        )
    )
    assert result.valid is True
    assert result.policy["planning"]["depth"] == "deep"
    assert result.provenance["planning.depth"] == "existing_default"
    assert result.policy["exploration"]["max_read_lines"] == 160
    assert result.provenance["exploration.max_read_lines"] == "skill_default"
    assert result.provenance["exploration.max_output_kb"] == "shadow_baseline"


def test_provider_ambiguity_falls_back_without_inventing_a_value() -> None:
    result = compile_task_policy(
        TaskPolicyContext(
            task_intent="maintenance",
            available_skills=(),
            existing_providers={
                "memory.max_capsules": (
                    PolicyProvider("existing_default", 1, 1),
                    PolicyProvider("existing_default", 2, 1),
                )
            },
        )
    )
    assert result.valid is False
    assert result.fallback is True
    assert result.policy is None
    assert result.reason_codes == ("provider_ambiguity",)


def test_policy_strings_are_bounded() -> None:
    result = compile_task_policy(
        TaskPolicyContext(task_intent="documentation", available_skills=())
    )
    assert result.valid is True
    assert validate_task_policy(
        {
            **result.policy,
            "planning": {"depth": "invented"},
        }
    ).reason_codes == ("invalid_enum:planning.depth",)


def test_kernel_fields_and_missing_skills_are_rejected() -> None:
    result = compile_task_policy(
        TaskPolicyContext(task_intent="feature", available_skills=())
    )
    policy = dict(result.policy)
    policy["action_gate"] = True
    assert validate_task_policy(policy).reason_codes == ("kernel_field_rejected",)

    policy = dict(result.policy)
    policy["capabilities"] = {"skills": ["backend-development"]}
    assert validate_task_policy(policy, available_skills=()).reason_codes == ("skill_missing",)


def test_same_input_and_registry_version_is_repeatable() -> None:
    context = TaskPolicyContext(
        task_intent="Fix a regression in the API",
        available_skills=AVAILABLE,
        formal_task_metadata={"task_class": "api"},
    )
    first = compile_task_policy(context)
    second = compile_task_policy(context)
    assert first == second
