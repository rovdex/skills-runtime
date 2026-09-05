from __future__ import annotations

import json

from shared_memory_runtime import (
    PolicyActivationResult,
    PolicyAdaptationResult,
    PolicyAdviceProvenance,
    PolicyCompileResult,
    build_jit_policy_evidence,
)


def _compiled() -> PolicyCompileResult:
    return PolicyCompileResult(
        policy={"task_class": "api", "secret": "must-not-leak"},
        valid=True,
        fallback=False,
        task_class="api",
        provenance={"task_class": "existing_default", "memory.scope": "skill_default"},
    )


def test_j2_typed_result_is_compiled_and_redacted() -> None:
    evidence = build_jit_policy_evidence(
        formal_task=True,
        compiler_result=_compiled(),
        metrics={"policy_valid": False, "policy_fallback": True, "policy_task_class": "wrong"},
        j2_executed=True,
    )

    assert evidence["j2"]["task_policy"] == "Compiled"
    assert evidence["j2"]["task_class"] == "api"
    assert evidence["j2"]["base_policy_source"] == "provider-derived"
    serialized = json.dumps(evidence, sort_keys=True)
    assert "must-not-leak" not in serialized
    assert "secret" not in serialized


def test_j2_fallback_and_invalid_typed_statuses_are_semantic() -> None:
    fallback = PolicyCompileResult({}, True, True, "general", provenance={"x": "shadow_baseline"})
    invalid = PolicyCompileResult(None, False, True, "general", ("provider_ambiguity",))

    assert build_jit_policy_evidence(formal_task=True, compiler_result=fallback)["j2"]["task_policy"] == "Fallback"
    assert build_jit_policy_evidence(formal_task=True, compiler_result=invalid)["j2"]["task_policy"] == "Invalid"


def test_stage_precedence_preserves_typed_result_and_reports_conflict() -> None:
    evidence = build_jit_policy_evidence(
        formal_task=True,
        compiler_result=_compiled(),
        j2_executed=False,
    )

    assert evidence["j2"]["task_policy"] == "Compiled"
    assert evidence["j2"]["evidence_consistency"] == "Conflict"
    assert evidence["evidence_consistency"] == "Conflict"
    assert evidence["evidence_consistency_conflicts"] == ["j2"]


def test_missing_typed_result_uses_explicit_execution_semantics() -> None:
    not_run = build_jit_policy_evidence(formal_task=True, j2_executed=False)
    unknown = build_jit_policy_evidence(formal_task=True, j2_executed=True)
    unavailable = build_jit_policy_evidence(formal_task=True)

    assert not_run["j2"]["task_policy"] == "Not Run"
    assert not_run["j2"]["reason"] == "not_requested"
    assert unknown["j2"]["task_policy"] == "Unknown"
    assert unknown["j2"]["reason"] == "evidence_missing_after_execution"
    assert unavailable["j2"]["task_policy"] == "Unknown"


def test_j2_metrics_fallback_requires_both_direct_status_fields() -> None:
    compiled = build_jit_policy_evidence(
        formal_task=True,
        metrics={"policy_valid": True, "policy_fallback": False, "policy_task_class": "feature"},
    )
    unknown = build_jit_policy_evidence(
        formal_task=True,
        metrics={"policy_valid": True, "policy_task_class": "feature"},
    )

    assert compiled["j2"]["task_policy"] == "Compiled"
    assert compiled["j2"]["task_class"] == "feature"
    assert unknown["j2"]["task_policy"] == "Unknown"


def test_j3_consumes_typed_adaptation_without_ranking() -> None:
    result = PolicyAdaptationResult(
        policy={"validation": {"full_suite": True}},
        adapted=True,
        source_experience_ids=("exp-ranked-first", "exp-ranked-second"),
        provenance=(
            PolicyAdviceProvenance(
                "validation.full_suite", False, True, "exp-ranked-first", "authoritative"
            ),
        ),
    )
    evidence = build_jit_policy_evidence(formal_task=True, adaptation_result=result)

    assert evidence["j3"]["experience_adaptation"] == "Applied"
    assert evidence["j3"]["adapted_fields"] == ["validation.full_suite"]
    assert evidence["j3"]["experience_policy_sources"] == ["exp-ranked-first", "exp-ranked-second"]
    assert evidence["j3"]["applied_advice"] == 1


def test_j3_no_advice_and_ambiguous_advice_are_distinct() -> None:
    no_advice = PolicyAdaptationResult({}, False, reason_codes=("no_eligible_advice",))
    ambiguous = PolicyAdaptationResult(
        {}, False, source_experience_ids=("exp-a", "exp-b"), reason_codes=("advice_winner_ambiguous",)
    )

    no_evidence = build_jit_policy_evidence(formal_task=True, adaptation_result=no_advice)
    conflict = build_jit_policy_evidence(formal_task=True, adaptation_result=ambiguous)
    assert no_evidence["j3"]["experience_adaptation"] == "No Eligible Advice"
    assert no_evidence["j3"]["eligible_experience_advice"] == 0
    assert conflict["j3"]["experience_adaptation"] == "Conflict Fallback"
    assert conflict["j3"]["experience_conflicts"] == 1


def test_j3_evaluated_no_change_and_missing_result() -> None:
    unchanged = PolicyAdaptationResult({}, False, source_experience_ids=("exp-a",))
    result = build_jit_policy_evidence(formal_task=True, adaptation_result=unchanged)
    missing = build_jit_policy_evidence(formal_task=True, j3_executed=True)

    assert result["j3"]["experience_adaptation"] == "Evaluated No Change"
    assert missing["j3"]["experience_adaptation"] == "Unknown"
    assert missing["j3"]["reason"] == "evidence_missing_after_execution"


def test_j4_protocol_only_and_runtime_enforced_fields_are_separate() -> None:
    result = PolicyActivationResult(
        active_policy={"planning": {"depth": "standard"}},
        fallback=False,
        policy_source="validated_jit_task_policy",
        observed_actual_behavior={"shell_calls": 2},
        protocol_consumed_fields=("planning.depth", "execution.shell_batching"),
        not_instrumented_fields=("planning.depth",),
        runtime_enforced_capabilities=("execution.shell_batching",),
    )
    evidence = build_jit_policy_evidence(formal_task=True, activation_result=result)

    assert evidence["j4"]["policy_activation"] == "Active"
    assert evidence["j4"]["policy_activation_enforcement"] == "Protocol Only"
    assert evidence["j4"]["host_enforced"] == "Not Guaranteed"
    assert evidence["j4"]["technically_unbypassable"] == "Not Guaranteed"
    assert evidence["j4"]["runtime_enforced_fields"] == ["execution.shell_batching"]
    assert evidence["j4"]["observed_compliance"] == "Partially Verified"


def test_j4_fallback_and_metrics_are_bounded() -> None:
    fallback = PolicyActivationResult(None, True, "existing_default")
    metrics = {
        "policy_mode": "active",
        "policy_observed_actual_behavior": {"file_reads": 4},
        "policy_protocol_consumed_fields": ("planning.depth",),
    }
    fallback_evidence = build_jit_policy_evidence(formal_task=True, activation_result=fallback)
    metrics_evidence = build_jit_policy_evidence(formal_task=True, metrics=metrics)

    assert fallback_evidence["j4"]["policy_activation"] == "Fallback"
    assert metrics_evidence["j4"]["policy_activation"] == "Active"
    assert metrics_evidence["j4"]["protocol_consumed_fields"] == ["planning.depth"]


def test_non_formal_interaction_has_no_lifecycle_disclosure() -> None:
    evidence = build_jit_policy_evidence(
        formal_task=False,
        compiler_result=_compiled(),
        j2_executed=True,
        j3_executed=True,
        j4_executed=True,
    )

    assert evidence["formal_task"] is False
    assert evidence["j2"]["task_policy"] == "Not Run"
    assert evidence["j2"]["reason"] == "non_formal_interaction"
    assert evidence["j3"]["reason"] == "non_formal_interaction"
    assert evidence["j4"]["reason"] == "non_formal_interaction"


def test_jit_evidence_is_deterministic_and_does_not_emit_policy_body() -> None:
    compiler = _compiled()
    first = build_jit_policy_evidence(
        formal_task=True,
        compiler_result=compiler,
        metrics={"policy_compile_ms": 1.5},
        j2_executed=True,
    )
    second = build_jit_policy_evidence(
        formal_task=True,
        compiler_result=compiler,
        metrics={"policy_compile_ms": 1.5},
        j2_executed=True,
    )

    assert first == second
    assert "policy" not in first["j2"]
    assert "prompt" not in json.dumps(first).lower()
    assert "response" not in json.dumps(first).lower()
    assert "cot" not in json.dumps(first).lower()
