"""Protocol-level JIT policy invocation for one Formal Task.

This module is an orchestration boundary, not a second policy engine.  It
invokes the existing J2 compiler, the existing J3 adaptation function over an
already-ranked Recall result, and the existing J4 activation function once
before the first normal task execution.  Host/Runner enforcement remains
outside this Runtime API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, TypeVar

from .jit_policy_evidence import build_jit_policy_evidence
from .jit_invocation_evidence import (
    JITInvocationEvidence,
    JITInvocationEvidenceConflict,
    JITInvocationEvidenceError,
    persist_jit_invocation_evidence,
)
from .policy_activation import PolicyActivationResult, activate_task_policy
from .policy_adaptation import PolicyAdaptationResult, SkillAdvice, adapt_task_policy
from .recall import RecallCandidate
from .task_policy import PolicyCompileResult, TaskPolicyContext, compile_task_policy


_ExecutionReturn = TypeVar("_ExecutionReturn")


@dataclass(frozen=True)
class FormalTaskJITResult:
    """The same-task typed evidence and activation hand-off."""

    task_id: str
    compiler_result: PolicyCompileResult
    adaptation_result: Optional[PolicyAdaptationResult]
    activation_result: Optional[PolicyActivationResult]
    execution_policy: Optional[Mapping[str, object]]
    evidence: Mapping[str, object]
    j2_executed: bool
    j3_executed: bool
    j4_executed: bool
    invocation_evidence: Optional[JITInvocationEvidence] = None

    @property
    def fallback_to_existing_behavior(self) -> bool:
        """Whether the caller must retain the existing default behavior."""

        return (
            self.compiler_result.fallback
            or (
                self.activation_result is not None
                and self.activation_result.fallback
            )
        )

    def evidence_for_task(self) -> Mapping[str, object]:
        """Return redacted evidence correlated with this Formal Task."""

        return {"task_id": self.task_id, **dict(self.evidence)}

    def metrics_fields(self) -> Mapping[str, object]:
        """Return only scalar/allowlisted fields for existing TaskMetrics."""

        j2 = self.evidence.get("j2", {})
        j3 = self.evidence.get("j3", {})
        j4 = self.evidence.get("j4", {})
        fields: Dict[str, object] = {
            "jit_policy_evidence_version": self.evidence.get("version"),
        }
        if isinstance(j2, Mapping):
            for name in (
                "policy_compile_ms",
                "task_class",
                "policy_valid",
                "policy_fallback",
            ):
                value = j2.get(name)
                if value not in {"Unknown", "Unavailable"}:
                    fields[
                        {
                            "task_class": "policy_task_class",
                            "policy_valid": "policy_valid",
                            "policy_fallback": "policy_fallback",
                        }.get(name, name)
                    ] = value
            if self.compiler_result.policy is not None:
                skills = self.compiler_result.policy.get("capabilities", {}).get("skills", [])
                if isinstance(skills, list):
                    fields["policy_skill_count"] = len(skills)
                for path, name in (
                    ("memory.max_capsules", "policy_max_capsules"),
                    ("exploration.max_read_lines", "policy_max_read_lines"),
                    ("exploration.max_output_kb", "policy_max_output_kb"),
                ):
                    value: object = self.compiler_result.policy
                    for part in path.split("."):
                        value = value.get(part) if isinstance(value, Mapping) else None
                    if isinstance(value, int) and not isinstance(value, bool):
                        fields[name] = value
        if isinstance(j3, Mapping):
            for source, target in (
                ("eligible_experience_advice", "experience_advice_count"),
                ("experience_conflicts", "experience_conflict_count"),
            ):
                value = j3.get(source)
                if isinstance(value, int) and not isinstance(value, bool):
                    fields[target] = value
            adapted_fields = j3.get("adapted_fields")
            if isinstance(adapted_fields, list):
                fields["adapted_field_count"] = len(adapted_fields)
            sources = j3.get("experience_policy_sources")
            if isinstance(sources, list):
                fields["experience_policy_sources"] = tuple(sources)
        if isinstance(j4, Mapping) and self.activation_result is not None:
            fields["policy_mode"] = "fallback" if self.activation_result.fallback else "active"
            fields["policy_source"] = self.activation_result.policy_source
            fields["policy_protocol_consumed_fields"] = tuple(
                j4.get("protocol_consumed_fields", ())
                if isinstance(j4.get("protocol_consumed_fields", ()), list)
                else ()
            )
            fields["policy_runtime_enforced_fields"] = tuple(
                j4.get("runtime_enforced_fields", ())
                if isinstance(j4.get("runtime_enforced_fields", ()), list)
                else ()
            )
            observed = self.activation_result.observed_actual_behavior
            fields["policy_observed_actual_behavior"] = dict(observed)
        return fields

    def consume_for_execution(
        self,
        execution_callback: Callable[[Optional[PolicyActivationResult]], _ExecutionReturn],
    ) -> _ExecutionReturn:
        """Pass the activation result to the first normal execution callback.

        The callback receives ``None`` only on the explicit existing-default
        fallback branch.  A valid J4 path always receives its typed
        ``PolicyActivationResult`` before execution begins.
        """

        if self.invocation_evidence is None:
            raise JITInvocationEvidenceError(
                "FormalTaskJITResult cannot be consumed before invocation evidence read-back"
            )
        return execution_callback(self.activation_result)


def prepare_formal_task_jit(
    task_id: str,
    *,
    context: TaskPolicyContext,
    ranked_candidates: Sequence[RecallCandidate] = (),
    advice_by_experience: Optional[
        Mapping[str, Optional[Mapping[str, object]]]
    ] = None,
    verified_skill_advice: Sequence[SkillAdvice] = (),
    skill_requirements: Optional[Mapping[str, object]] = None,
    project_requirements: Optional[Mapping[str, object]] = None,
    user_requirements: Optional[Mapping[str, object]] = None,
    actual_metrics: Optional[Mapping[str, object]] = None,
    evidence_metrics: Optional[Mapping[str, object]] = None,
    runtime_enforced_fields: Sequence[str] = (),
    codex_home: Optional[Path] = None,
) -> FormalTaskJITResult:
    """Prepare one Formal Task's J2/J3/J4 policy before normal execution.

    ``ranked_candidates`` must already be the authoritative Recall ordering;
    this function deliberately performs no ranking or applicability search.
    ``task_id`` is correlation evidence only and does not alter Runtime State.
    """

    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a non-empty string")
    if not isinstance(context, TaskPolicyContext):
        raise TypeError("context must be a TaskPolicyContext")

    started_ns = time.perf_counter_ns()
    compiler_result = compile_task_policy(context)
    compile_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    metrics = dict(evidence_metrics or {})
    metrics.setdefault("policy_compile_ms", compile_ms)
    if not compiler_result.valid or compiler_result.policy is None:
        evidence = build_jit_policy_evidence(
            formal_task=True,
            compiler_result=compiler_result,
            metrics=metrics,
            j2_executed=True,
            j3_executed=False,
            j4_executed=False,
        )
        result = FormalTaskJITResult(
            task_id=task_id,
            compiler_result=compiler_result,
            adaptation_result=None,
            activation_result=None,
            execution_policy=None,
            evidence=evidence,
            j2_executed=True,
            j3_executed=False,
            j4_executed=False,
        )
        return _persist_and_attach_invocation_evidence(result, codex_home=codex_home)

    adaptation_result = adapt_task_policy(
        compiler_result.policy,
        tuple(ranked_candidates),
        advice_by_experience or {},
        available_skills=context.available_skills,
        verified_skill_advice=verified_skill_advice,
    )
    activation_result = activate_task_policy(
        adaptation_result.policy,
        available_skills=context.available_skills,
        skill_requirements=skill_requirements,
        project_requirements=project_requirements,
        user_requirements=user_requirements,
        actual_metrics=actual_metrics,
        runtime_enforced_fields=runtime_enforced_fields,
    )
    evidence = build_jit_policy_evidence(
        formal_task=True,
        compiler_result=compiler_result,
        adaptation_result=adaptation_result,
        activation_result=activation_result,
        metrics=metrics,
        j2_executed=True,
        j3_executed=True,
        j4_executed=True,
    )
    result = FormalTaskJITResult(
        task_id=task_id,
        compiler_result=compiler_result,
        adaptation_result=adaptation_result,
        activation_result=activation_result,
        execution_policy=activation_result.active_policy,
        evidence=evidence,
        j2_executed=True,
        j3_executed=True,
        j4_executed=True,
    )
    return _persist_and_attach_invocation_evidence(result, codex_home=codex_home)


def _persist_and_attach_invocation_evidence(
    result: FormalTaskJITResult, *, codex_home: Optional[Path]
) -> FormalTaskJITResult:
    persisted = persist_jit_invocation_evidence(result, codex_home=codex_home)
    if persisted.status == "conflict":
        raise JITInvocationEvidenceConflict(
            persisted.reason or "semantic_evidence_conflict"
        )
    if persisted.status not in {"created", "reused"} or persisted.evidence is None:
        reason = persisted.reason or "invocation_evidence_persist_failed"
        raise JITInvocationEvidenceError(reason)
    return replace(result, invocation_evidence=persisted.evidence)


__all__ = ["FormalTaskJITResult", "prepare_formal_task_jit"]
