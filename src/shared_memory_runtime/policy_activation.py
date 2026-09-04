"""J4 protocol-only Task Policy activation and evidence classification."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .policy_adaptation import validate_policy_advice
from .task_policy import PolicyValidation, validate_task_policy


OBSERVED_ACTUAL_FIELDS = frozenset(
    {"shell_calls", "wsl_process_count", "file_reads", "tool_output_bytes", "capsule_count"}
)

PROTOCOL_ONLY_FIELDS = frozenset(
    {
        "planning.depth",
        "capabilities.skills",
        "memory.scope",
        "memory.max_capsules",
        "memory.full_markdown",
        "exploration.strategy",
        "exploration.max_read_lines",
        "exploration.max_output_kb",
        "execution.shell_batching",
        "execution.wsl_batching",
        "execution.parallel_exploration",
        "validation.focused_tests",
        "validation.full_suite",
        "agent_execution_strategy",
    }
)


@dataclass(frozen=True)
class PolicyActivationResult:
    active_policy: Optional[Mapping[str, object]]
    fallback: bool
    policy_source: str
    observed_actual_behavior: Mapping[str, int] = field(default_factory=dict)
    observed_compliance: Mapping[str, str] = field(default_factory=dict)
    protocol_consumed_fields: Tuple[str, ...] = ()
    not_instrumented_fields: Tuple[str, ...] = ()
    runtime_enforced_capabilities: Tuple[str, ...] = ()
    reason_codes: Tuple[str, ...] = ()

    @property
    def policy_activation_enforcement(self) -> str:
        return "Protocol Only"

    @property
    def host_enforced(self) -> bool:
        return False

    @property
    def technically_unbypassable(self) -> bool:
        return False


def _flatten_paths(value: Mapping[str, object], prefix: str = "") -> Iterable[Tuple[str, object]]:
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            yield from _flatten_paths(child, path)
        else:
            yield path, child


def _set_path(target: Dict[str, object], path: str, value: object) -> None:
    cursor = target
    parts = path.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _apply_requirements(
    policy: Mapping[str, object],
    requirements: Mapping[str, object],
    *,
    available_skills: Sequence[str],
) -> Tuple[Optional[Mapping[str, object]], Optional[str]]:
    validation = validate_policy_advice(requirements, available_skills=available_skills)
    if not validation.valid:
        return None, validation.reason_codes[0]
    merged = deepcopy(dict(policy))
    for path, value in _flatten_paths(requirements):
        _set_path(merged, path, value)
    final_validation = validate_task_policy(merged, available_skills=available_skills)
    if not final_validation.valid:
        return None, final_validation.reason_codes[0]
    return merged, None


def _observations(actual_metrics: Optional[Mapping[str, object]]) -> Tuple[Mapping[str, int], Tuple[str, ...]]:
    if not actual_metrics:
        return {}, ()
    observed: Dict[str, int] = {}
    reasons = []
    for key, value in actual_metrics.items():
        if key not in OBSERVED_ACTUAL_FIELDS:
            reasons.append("unrecognized_actual_metric")
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            reasons.append(f"invalid_actual_metric:{key}")
            continue
        observed[key] = value
    return observed, tuple(sorted(set(reasons)))


def activate_task_policy(
    policy: Mapping[str, object],
    *,
    available_skills: Sequence[str] = (),
    skill_requirements: Optional[Mapping[str, object]] = None,
    project_requirements: Optional[Mapping[str, object]] = None,
    user_requirements: Optional[Mapping[str, object]] = None,
    actual_metrics: Optional[Mapping[str, object]] = None,
    runtime_enforced_fields: Sequence[str] = (),
) -> PolicyActivationResult:
    """Consume a validated policy through protocol-only activation.

    Requirement precedence is Skill safety/default bounds, authoritative
    Project requirements, then explicit current user requirements. No input
    can control the Formal Task Kernel. Host enforcement remains false.
    """

    initial = validate_task_policy(policy, available_skills=available_skills)
    if not initial.valid:
        observed, observation_reasons = _observations(actual_metrics)
        return PolicyActivationResult(
            active_policy=None,
            fallback=True,
            policy_source="existing_default",
            observed_actual_behavior=observed,
            reason_codes=("policy_invalid",) + initial.reason_codes + observation_reasons,
        )

    active: Mapping[str, object] = policy
    source = "validated_jit_task_policy"
    reasons = []
    for name, requirements in (
        ("skill_default", skill_requirements),
        ("project_requirement", project_requirements),
        ("user_requirement", user_requirements),
    ):
        if requirements is None:
            continue
        active, error = _apply_requirements(active, requirements, available_skills=available_skills)
        if error:
            observed, observation_reasons = _observations(actual_metrics)
            return PolicyActivationResult(
                active_policy=None,
                fallback=True,
                policy_source="existing_default",
                observed_actual_behavior=observed,
                reason_codes=("activation_requirement_invalid", error) + observation_reasons,
            )
        source = name

    observed, observation_reasons = _observations(actual_metrics)
    policy_paths = tuple(path for path, _ in _flatten_paths(active))
    protocol_fields = tuple(sorted(set(policy_paths) & PROTOCOL_ONLY_FIELDS))
    runtime_fields = tuple(sorted(set(runtime_enforced_fields) & set(protocol_fields)))
    not_instrumented = tuple(path for path in protocol_fields if path not in runtime_fields)
    compliance = {key: "observed" for key in observed}
    compliance.update({key: "not_independently_instrumented" for key in not_instrumented})
    reasons.extend(observation_reasons)
    return PolicyActivationResult(
        active_policy=active,
        fallback=False,
        policy_source=source,
        observed_actual_behavior=observed,
        observed_compliance=compliance,
        protocol_consumed_fields=protocol_fields,
        not_instrumented_fields=not_instrumented,
        runtime_enforced_capabilities=runtime_fields,
        reason_codes=tuple(sorted(set(reasons))),
    )


__all__ = [
    "OBSERVED_ACTUAL_FIELDS",
    "PROTOCOL_ONLY_FIELDS",
    "PolicyActivationResult",
    "activate_task_policy",
]
