"""J3 Experience Advice overlay over an already ranked Recall result."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .recall import RecallCandidate
from .task_policy import (
    KERNEL_FIELDS,
    NUMERIC_BOUNDS,
    POLICY_ENUMS,
    PolicyValidation,
    validate_task_policy,
)


@dataclass(frozen=True)
class PolicyAdvice:
    experience_id: str
    fields: Mapping[str, object]


@dataclass(frozen=True)
class PolicyAdviceProvenance:
    field: str
    base_value: object
    final_value: object
    source_experience_id: str
    reason: str


@dataclass(frozen=True)
class PolicyAdaptationResult:
    policy: Mapping[str, object]
    adapted: bool
    source_experience_ids: Tuple[str, ...] = ()
    reason_codes: Tuple[str, ...] = ()
    provenance: Tuple[PolicyAdviceProvenance, ...] = ()


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


def _get_path(value: Mapping[str, object], path: str) -> object:
    cursor: object = value
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def validate_policy_advice(
    advice: Mapping[str, object], *, available_skills: Sequence[str] = ()
) -> PolicyValidation:
    """Validate a partial advice mapping against the J2 whitelist."""

    if not isinstance(advice, Mapping):
        return PolicyValidation(False, ("advice_not_mapping",))
    for path, value in _flatten_paths(advice):
        if path in KERNEL_FIELDS or path.split(".")[-1] in KERNEL_FIELDS:
            return PolicyValidation(False, ("kernel_advice_rejected",))
        if path not in {
            "task_class",
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
        }:
            return PolicyValidation(False, ("unknown_advice_field",))
        if path == "capabilities.skills":
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                return PolicyValidation(False, ("invalid_type:capabilities.skills",))
        elif path in NUMERIC_BOUNDS:
            if not isinstance(value, int) or isinstance(value, bool):
                return PolicyValidation(False, (f"invalid_type:{path}",))
        elif path in {"memory.full_markdown", "execution.shell_batching", "execution.wsl_batching", "execution.parallel_exploration", "validation.focused_tests", "validation.full_suite"}:
            if not isinstance(value, bool):
                return PolicyValidation(False, (f"invalid_type:{path}",))
        elif not isinstance(value, str):
            return PolicyValidation(False, (f"invalid_type:{path}",))
        if path in POLICY_ENUMS and value not in POLICY_ENUMS[path]:
            return PolicyValidation(False, (f"invalid_enum:{path}",))
        if path in NUMERIC_BOUNDS:
            minimum, maximum = NUMERIC_BOUNDS[path]
            if not minimum <= value <= maximum:
                return PolicyValidation(False, (f"invalid_bounds:{path}",))
    skills = advice.get("capabilities", {})
    if isinstance(skills, Mapping) and "skills" in skills:
        values = skills["skills"]
        if len(set(values)) != len(values):
            return PolicyValidation(False, ("duplicate_skill",))
        if set(values) - set(available_skills):
            return PolicyValidation(False, ("skill_missing",))
    return PolicyValidation(True)


def adapt_task_policy(
    base_policy: Mapping[str, object],
    ranked_candidates: Sequence[RecallCandidate],
    advice_by_experience: Mapping[str, Optional[Mapping[str, object]]],
    *,
    available_skills: Sequence[str] = (),
) -> PolicyAdaptationResult:
    """Apply only the first unique advice value in authoritative rank order.

    This function deliberately does not sort, score, inspect anchors, or
    compare project applicability.  ``ranked_candidates`` is already the
    result of the Runtime's authoritative Recall ranking.
    """

    eligible = []
    reasons = []
    for candidate in ranked_candidates:
        if (
            candidate.effective_experience_verification != "verified"
            or not candidate.remote_verified
            or not candidate.applicability_passed
        ):
            continue
        raw_advice = advice_by_experience.get(candidate.experience_id)
        if raw_advice is None:
            continue
        validation = validate_policy_advice(raw_advice, available_skills=available_skills)
        if not validation.valid:
            reasons.extend(validation.reason_codes)
            continue
        eligible.append((candidate, raw_advice))

    if not eligible:
        return PolicyAdaptationResult(base_policy, False, reason_codes=tuple(sorted(set(reasons or ["no_eligible_advice"]))))

    winner, winner_advice = eligible[0]
    for other, other_advice in eligible[1:]:
        if other.score == winner.score and other_advice != winner_advice:
            return PolicyAdaptationResult(
                base_policy,
                False,
                source_experience_ids=tuple(item[0].experience_id for item in eligible),
                reason_codes=("advice_winner_ambiguous",),
            )

    adapted = deepcopy(dict(base_policy))
    provenance = []
    for path, value in _flatten_paths(winner_advice):
        base_value = _get_path(adapted, path)
        _set_path(adapted, path, value)
        provenance.append(
            PolicyAdviceProvenance(
                field=path,
                base_value=base_value,
                final_value=value,
                source_experience_id=winner.experience_id,
                reason="authoritative_ranked_experience_advice",
            )
        )
    final_validation = validate_task_policy(adapted, available_skills=available_skills)
    if not final_validation.valid:
        return PolicyAdaptationResult(
            base_policy,
            False,
            source_experience_ids=(winner.experience_id,),
            reason_codes=("adapted_policy_invalid",) + final_validation.reason_codes,
        )
    return PolicyAdaptationResult(
        policy=adapted,
        adapted=True,
        source_experience_ids=(winner.experience_id,),
        reason_codes=tuple(sorted(set(reasons))),
        provenance=tuple(provenance),
    )


__all__ = [
    "PolicyAdvice",
    "PolicyAdviceProvenance",
    "PolicyAdaptationResult",
    "adapt_task_policy",
    "validate_policy_advice",
]
