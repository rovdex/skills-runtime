"""Read-only disclosure of deterministic JIT policy evidence.

This module is deliberately a serializer boundary.  It consumes the typed
results produced by J2, J3, and J4 and never runs compilation, ranking,
applicability, activation, persistence, or model work itself.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple, Union


JIT_POLICY_EVIDENCE_VERSION = "jit-policy-evidence-v1"
_UNKNOWN = "Unknown"
_UNAVAILABLE = "Unavailable"
_CONSISTENT = "Consistent"
_CONFLICT = "Conflict"
_MISSING = object()


def _metrics_mapping(metrics: object) -> Mapping[str, object]:
    if metrics is None:
        return {}
    if isinstance(metrics, Mapping):
        return metrics
    as_mapping = getattr(metrics, "as_mapping", None)
    if callable(as_mapping):
        value = as_mapping()
        if isinstance(value, Mapping):
            return value
    return {}


def _safe_reason_codes(value: object) -> Tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        return ()
    return tuple(sorted({item for item in value if isinstance(item, str) and item}))


def _safe_string(value: object, default: object = _UNKNOWN) -> object:
    return value if isinstance(value, str) and value else default


def _safe_bool(value: object) -> object:
    return value if isinstance(value, bool) else _UNKNOWN


def _safe_non_negative_number(value: object, default: object = _UNAVAILABLE) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return default
    return value


def _safe_non_negative_int(value: object, default: object = _UNKNOWN) -> object:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _safe_string_sequence(value: object, *, empty_as_none: bool = False) -> object:
    if not isinstance(value, (tuple, list)):
        return _UNKNOWN
    values = sorted({item for item in value if isinstance(item, str) and item})
    if empty_as_none and not values:
        return None
    return values


def _ordered_string_sequence(value: object) -> object:
    if not isinstance(value, (tuple, list)):
        return _UNKNOWN
    values = []
    seen = set()
    for item in value:
        if isinstance(item, str) and item and item not in seen:
            seen.add(item)
            values.append(item)
    return values


def _typed_stage_state(
    typed_result: object,
    executed: Optional[bool],
    *,
    metrics_status: Optional[str] = None,
) -> Tuple[str, Optional[str], str]:
    """Return ``(status, reason, consistency)`` using the fixed precedence."""

    if typed_result is not None:
        return "executed", None, _CONFLICT if executed is False else _CONSISTENT
    if executed is False:
        return "not_run", "not_requested", _CONSISTENT
    if executed is True:
        return "unknown", "evidence_missing_after_execution", _CONSISTENT
    if metrics_status is not None:
        return "metrics", None, _CONSISTENT
    return "unknown", "evidence_unavailable", _CONSISTENT


def _result_base(status: str, reason: Optional[str], consistency: str) -> Dict[str, object]:
    result: Dict[str, object] = {"evidence_consistency": consistency}
    if reason is not None:
        result["reason"] = reason
    return result


def _j2_metrics_status(metrics: Mapping[str, object]) -> Optional[str]:
    valid = metrics.get("policy_valid", _MISSING)
    fallback = metrics.get("policy_fallback", _MISSING)
    if not isinstance(valid, bool) or not isinstance(fallback, bool):
        return None
    if not valid:
        return "Invalid"
    return "Fallback" if fallback else "Compiled"


def _j2_evidence(
    compiler_result: object,
    metrics: Mapping[str, object],
    executed: Optional[bool],
) -> Dict[str, object]:
    metrics_status = _j2_metrics_status(metrics)
    precedence, reason, consistency = _typed_stage_state(
        compiler_result, executed, metrics_status=metrics_status
    )
    if precedence == "executed":
        valid = getattr(compiler_result, "valid", _MISSING)
        fallback = getattr(compiler_result, "fallback", _MISSING)
        if valid is False:
            status = "Invalid"
        elif valid is True and fallback is True:
            status = "Fallback"
        elif valid is True and fallback is False:
            status = "Compiled"
        else:
            status = _UNKNOWN
            reason = "typed_result_incomplete"
        task_class = _safe_string(getattr(compiler_result, "task_class", _MISSING))
        policy_valid = _safe_bool(valid)
        policy_fallback = _safe_bool(fallback)
        provenance = getattr(compiler_result, "provenance", {})
        if isinstance(provenance, Mapping):
            sources = {value for value in provenance.values() if isinstance(value, str)}
            if fallback is True:
                base_source: object = "fallback"
            elif sources.intersection({"existing_default", "skill_default"}):
                base_source = "provider-derived"
            elif "shadow_baseline" in sources:
                base_source = "shadow_baseline"
            elif sources:
                base_source = "actual"
            else:
                base_source = _UNKNOWN
        else:
            base_source = "fallback" if fallback is True else _UNKNOWN
    elif precedence == "metrics":
        status = metrics_status or _UNKNOWN
        task_class = _safe_string(metrics.get("policy_task_class"))
        policy_valid = _safe_bool(metrics.get("policy_valid"))
        policy_fallback = _safe_bool(metrics.get("policy_fallback"))
        base_source = _safe_string(metrics.get("policy_source"))
        if base_source not in {"existing_default", "skill_default", "shadow_baseline", "fallback", "actual"}:
            base_source = _UNKNOWN
        elif base_source in {"existing_default", "skill_default"}:
            base_source = "provider-derived"
    else:
        status = "Not Run" if precedence == "not_run" else _UNKNOWN
        task_class = _UNKNOWN
        policy_valid = _UNKNOWN
        policy_fallback = _UNKNOWN
        base_source = _UNKNOWN

    result = _result_base(reason=reason, status=status, consistency=consistency)
    result.update(
        {
            "task_policy": status,
            "task_class": task_class,
            "policy_valid": policy_valid,
            "policy_fallback": policy_fallback,
            "policy_compile_ms": _safe_non_negative_number(metrics.get("policy_compile_ms")),
            "base_policy_source": base_source,
        }
    )
    if precedence == "not_run":
        result["task_policy"] = "Not Run"
    return result


def _changed_advice_fields(adaptation_result: object) -> Tuple[str, ...]:
    provenance = getattr(adaptation_result, "provenance", ())
    if not isinstance(provenance, (tuple, list)):
        return ()
    fields = {
        item.field
        for item in provenance
        if isinstance(getattr(item, "field", None), str)
        and getattr(item, "base_value", _MISSING) != getattr(item, "final_value", _MISSING)
    }
    return tuple(sorted(fields))


def _j3_evidence(
    adaptation_result: object,
    metrics: Mapping[str, object],
    executed: Optional[bool],
) -> Dict[str, object]:
    precedence, reason, consistency = _typed_stage_state(adaptation_result, executed)
    if precedence == "executed":
        reason_codes = _safe_reason_codes(getattr(adaptation_result, "reason_codes", ()))
        source_ids = _ordered_string_sequence(
            getattr(adaptation_result, "source_experience_ids", ())
        )
        if not isinstance(source_ids, list):
            source_ids = []
        changed_fields = list(_changed_advice_fields(adaptation_result))
        adapted = getattr(adaptation_result, "adapted", _MISSING)
        if "advice_winner_ambiguous" in reason_codes:
            status = "Conflict Fallback"
        elif "no_eligible_advice" in reason_codes and not source_ids:
            status = "No Eligible Advice"
        elif adapted is True and changed_fields:
            status = "Applied"
        elif adapted is False and source_ids:
            status = "Evaluated No Change"
        elif adapted is True:
            status = "Evaluated No Change"
        elif not source_ids:
            status = "No Eligible Advice"
        else:
            status = _UNKNOWN
            reason = "typed_result_incomplete"
        advice_count = _safe_non_negative_int(metrics.get("experience_advice_count"))
        if advice_count == _UNKNOWN and "no_eligible_advice" in reason_codes and not source_ids:
            advice_count = 0
        applied_count: object = len(
            {
                getattr(item, "source_experience_id")
                for item in getattr(adaptation_result, "provenance", ())
                if getattr(item, "source_experience_id", None) in source_ids
                and getattr(item, "base_value", _MISSING) != getattr(item, "final_value", _MISSING)
            }
        )
        conflict_count = _safe_non_negative_int(metrics.get("experience_conflict_count"))
        if conflict_count == _UNKNOWN:
            conflict_count = 1 if "advice_winner_ambiguous" in reason_codes else 0
        result = _result_base(reason=reason, status=status, consistency=consistency)
        result.update(
            {
                "experience_adaptation": status,
                "eligible_experience_advice": advice_count,
                "applied_advice": applied_count,
                "adapted_fields": changed_fields,
                "experience_conflicts": conflict_count,
                "experience_policy_sources": source_ids,
            }
        )
    elif precedence == "not_run":
        result = _result_base(reason=reason, status="Not Run", consistency=consistency)
        result.update(
            {
                "experience_adaptation": "Not Run",
                "eligible_experience_advice": _UNKNOWN,
                "applied_advice": _UNKNOWN,
                "adapted_fields": _UNKNOWN,
                "experience_conflicts": _UNKNOWN,
                "experience_policy_sources": _UNKNOWN,
            }
        )
    else:
        result = _result_base(reason=reason, status=_UNKNOWN, consistency=consistency)
        result.update(
            {
                "experience_adaptation": _UNKNOWN,
                "eligible_experience_advice": _UNKNOWN,
                "applied_advice": _UNKNOWN,
                "adapted_fields": _UNKNOWN,
                "experience_conflicts": _UNKNOWN,
                "experience_policy_sources": _UNKNOWN,
            }
        )
    return result


def _metric_string_sequence(metrics: Mapping[str, object], key: str) -> object:
    if key not in metrics:
        return _UNKNOWN
    return _safe_string_sequence(metrics[key], empty_as_none=True)


def _j4_metrics_status(metrics: Mapping[str, object]) -> Optional[str]:
    mode = metrics.get("policy_mode", _MISSING)
    if mode == "active":
        return "Active"
    if mode == "fallback":
        return "Fallback"
    return None


def _j4_observed_compliance(
    observed: object,
    not_instrumented: object,
) -> str:
    has_observed = isinstance(observed, Mapping) and bool(observed)
    has_uninstrumented = isinstance(not_instrumented, (tuple, list)) and bool(not_instrumented)
    if has_observed and has_uninstrumented:
        return "Partially Verified"
    if has_observed:
        return "Verified"
    if has_uninstrumented:
        return "Not Independently Instrumented"
    return _UNKNOWN


def _j4_evidence(
    activation_result: object,
    metrics: Mapping[str, object],
    executed: Optional[bool],
) -> Dict[str, object]:
    metrics_status = _j4_metrics_status(metrics)
    precedence, reason, consistency = _typed_stage_state(
        activation_result, executed, metrics_status=metrics_status
    )
    if precedence == "executed":
        fallback = getattr(activation_result, "fallback", _MISSING)
        active_policy = getattr(activation_result, "active_policy", _MISSING)
        if fallback is True:
            status = "Fallback"
        elif fallback is False and active_policy is not None and active_policy is not _MISSING:
            status = "Active"
        else:
            status = _UNKNOWN
            reason = "typed_result_incomplete"
        applied = _metric_string_sequence(metrics, "policy_applied_fields")
        runtime_fields = _safe_string_sequence(
            getattr(activation_result, "runtime_enforced_capabilities", _MISSING),
            empty_as_none=True,
        )
        protocol_fields = _safe_string_sequence(
            getattr(activation_result, "protocol_consumed_fields", _MISSING),
            empty_as_none=True,
        )
        if runtime_fields == _UNKNOWN and "policy_runtime_enforced_fields" in metrics:
            runtime_fields = _metric_string_sequence(metrics, "policy_runtime_enforced_fields")
        if protocol_fields == _UNKNOWN and "policy_protocol_consumed_fields" in metrics:
            protocol_fields = _metric_string_sequence(metrics, "policy_protocol_consumed_fields")
        observed = getattr(activation_result, "observed_actual_behavior", _MISSING)
        not_instrumented = getattr(activation_result, "not_instrumented_fields", _MISSING)
        observed_compliance = _j4_observed_compliance(observed, not_instrumented)
        reason_codes = _safe_reason_codes(getattr(activation_result, "reason_codes", ()))
    elif precedence == "metrics":
        status = metrics_status or _UNKNOWN
        applied = _metric_string_sequence(metrics, "policy_applied_fields")
        runtime_fields = _metric_string_sequence(metrics, "policy_runtime_enforced_fields")
        protocol_fields = _metric_string_sequence(metrics, "policy_protocol_consumed_fields")
        observed = metrics.get("policy_observed_actual_behavior", _MISSING)
        not_instrumented = metrics.get("policy_not_instrumented_fields", _MISSING)
        observed_compliance = _j4_observed_compliance(observed, not_instrumented)
        reason_codes = ()
    elif precedence == "not_run":
        status = "Not Run"
        applied = runtime_fields = protocol_fields = _UNKNOWN
        observed_compliance = _UNKNOWN
        reason_codes = ()
    else:
        status = _UNKNOWN
        applied = runtime_fields = protocol_fields = _UNKNOWN
        observed_compliance = _UNKNOWN
        reason_codes = ()

    result = _result_base(reason=reason, status=status, consistency=consistency)
    result.update(
        {
            "policy_activation": status,
            "policy_activation_enforcement": "Protocol Only",
            "host_enforced": "Not Guaranteed",
            "technically_unbypassable": "Not Guaranteed",
            "applied_policy_fields": applied,
            "runtime_enforced_fields": runtime_fields,
            "protocol_consumed_fields": protocol_fields,
            "observed_compliance": observed_compliance,
        }
    )
    if reason_codes:
        result["reason_codes"] = list(reason_codes)
    return result


def build_jit_policy_evidence(
    *,
    formal_task: bool,
    compiler_result: Optional[object] = None,
    adaptation_result: Optional[object] = None,
    activation_result: Optional[object] = None,
    metrics: Optional[Union[Mapping[str, object], object]] = None,
    j2_executed: Optional[bool] = None,
    j3_executed: Optional[bool] = None,
    j4_executed: Optional[bool] = None,
) -> Mapping[str, object]:
    """Build a deterministic, redacted J2/J3/J4 evidence mapping.

    ``formal_task`` is explicit so ordinary interactions cannot accidentally
    acquire lifecycle disclosure.  The function does not mutate any input.
    """

    if not isinstance(formal_task, bool):
        raise TypeError("formal_task must be a bool")
    if not formal_task:
        non_formal = {
            "evidence_consistency": _CONSISTENT,
            "evidence_consistency_conflicts": [],
            "formal_task": False,
            "j2": {
                "task_policy": "Not Run",
                "reason": "non_formal_interaction",
                "evidence_consistency": _CONSISTENT,
            },
            "j3": {
                "experience_adaptation": "Not Run",
                "reason": "non_formal_interaction",
                "evidence_consistency": _CONSISTENT,
            },
            "j4": {
                "policy_activation": "Not Run",
                "reason": "non_formal_interaction",
                "evidence_consistency": _CONSISTENT,
            },
        }
        return {"version": JIT_POLICY_EVIDENCE_VERSION, **non_formal}

    metric_values = _metrics_mapping(metrics)
    j2 = _j2_evidence(compiler_result, metric_values, j2_executed)
    j3 = _j3_evidence(adaptation_result, metric_values, j3_executed)
    j4 = _j4_evidence(activation_result, metric_values, j4_executed)
    conflicts = [
        stage
        for stage, value in (("j2", j2), ("j3", j3), ("j4", j4))
        if value.get("evidence_consistency") == _CONFLICT
    ]
    return {
        "version": JIT_POLICY_EVIDENCE_VERSION,
        "formal_task": True,
        "evidence_consistency": _CONFLICT if conflicts else _CONSISTENT,
        "evidence_consistency_conflicts": conflicts,
        "j2": j2,
        "j3": j3,
        "j4": j4,
    }


__all__ = ["JIT_POLICY_EVIDENCE_VERSION", "build_jit_policy_evidence"]
