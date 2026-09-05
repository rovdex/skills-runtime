"""Durable, write-once evidence for one Formal Task JIT preparation.

The artifact is deliberately smaller than a policy or an execution record. It
stores only the allowlisted semantic facts needed to audit the J2/J3/J4
invocation after the in-memory execution context is gone. The artifact is
local-only and is not Finalization State, Shared Memory, or a lifecycle store.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from .task_policy import TASK_CLASSES


JIT_INVOCATION_EVIDENCE_VERSION = "jit-invocation-evidence-v1"
_TASK_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ACTIVATION_SOURCES = frozenset(
    {
        "existing_default",
        "validated_jit_task_policy",
        "skill_default",
        "project_requirement",
        "user_requirement",
    }
)
_J3_STATUSES = frozenset(
    {"Applied", "Conflict Fallback", "Evaluated No Change", "No Eligible Advice", "Not Run", "Unknown", "Unavailable"}
)
_J4_STATUSES = frozenset({"Active", "Fallback", "Not Run", "Unknown", "Unavailable"})
_POLICY_FIELDS = (
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
)
_UNORDERED_POLICY_FIELDS = frozenset(
    {"capabilities.skills", "policy_applied_fields", "runtime_enforced_fields", "protocol_consumed_fields"}
)
_EVIDENCE_FIELDS = (
    "task_id",
    "evidence_version",
    "prepared_at",
    "policy_task_class",
    "policy_valid",
    "policy_fallback",
    "policy_compile_ms",
    "base_policy_hash",
    "adapted_policy_hash",
    "activation_result_hash",
    "j3_status",
    "experience_advice_count",
    "adapted_field_count",
    "experience_conflict_count",
    "experience_policy_sources",
    "j4_status",
    "policy_applied_fields",
    "runtime_enforced_fields",
    "protocol_consumed_fields",
    "activation_enforcement",
)
_SEMANTIC_EVIDENCE_FIELDS = tuple(
    field for field in _EVIDENCE_FIELDS if field not in {"prepared_at", "policy_compile_ms"}
)
_ALLOWED_ACTIVATION_REASON_CODES = frozenset(
    {
        "policy_invalid",
        "activation_requirement_invalid",
        "policy_not_mapping",
        "kernel_field_rejected",
        "unknown_policy_field",
        "advice_not_mapping",
        "kernel_advice_rejected",
        "unknown_advice_field",
        "invalid_type",
        "invalid_enum",
        "invalid_bounds",
        "duplicate_skill",
        "skill_missing",
        "unrecognized_actual_metric",
        "invalid_actual_metric",
    }
)


class JITInvocationEvidenceError(RuntimeError):
    """The checkpoint could not be safely created or validated."""


class JITInvocationEvidenceConflict(JITInvocationEvidenceError):
    """Existing or cross-source evidence disagrees with the candidate."""


class JITInvocationEvidenceInvalid(JITInvocationEvidenceError):
    """Existing evidence is corrupt or violates the artifact contract."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise JITInvocationEvidenceInvalid("value is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _get_path(value: Mapping[str, object], path: str) -> object:
    current: object = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _set_path(target: Dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    cursor = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _canonical_unordered(value: object, field: str) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise JITInvocationEvidenceInvalid(f"{field} must be a collection")
    values = []
    seen = set()
    for item in value:
        if not isinstance(item, str) or not item:
            raise JITInvocationEvidenceInvalid(f"{field} must contain non-empty strings")
        if item in seen:
            raise JITInvocationEvidenceInvalid(f"{field} contains duplicates")
        seen.add(item)
        values.append(item)
    return tuple(sorted(values))


def _canonical_ordered(value: object, field: str) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise JITInvocationEvidenceInvalid(f"{field} must be a collection")
    values = []
    seen = set()
    for item in value:
        if not isinstance(item, str) or not item:
            raise JITInvocationEvidenceInvalid(f"{field} must contain non-empty strings")
        if item in seen:
            raise JITInvocationEvidenceInvalid(f"{field} contains duplicates")
        seen.add(item)
        values.append(item)
    return tuple(values)


def canonical_policy_serialization(policy: Optional[Mapping[str, object]]) -> Optional[Mapping[str, object]]:
    """Return only the deterministic, whitelisted semantic Policy shape."""

    if policy is None:
        return None
    if not isinstance(policy, Mapping):
        raise JITInvocationEvidenceInvalid("Policy must be a mapping")
    result: Dict[str, object] = {}
    for path in _POLICY_FIELDS:
        value = _get_path(policy, path)
        if path == "capabilities.skills":
            value = _canonical_unordered(value, path)
            if value is not None:
                value = list(value)
        _set_path(result, path, value)
    return result


def policy_semantic_hash(policy: Optional[Mapping[str, object]]) -> Optional[str]:
    """Hash a Policy without persisting or hashing its non-whitelisted body."""

    canonical = canonical_policy_serialization(policy)
    return _sha256(canonical) if canonical is not None else None


def _canonical_activation_serialization(activation_result: object) -> Optional[Mapping[str, object]]:
    if activation_result is None:
        return None
    active_policy = getattr(activation_result, "active_policy", None)
    fallback = getattr(activation_result, "fallback", None)
    policy_source = getattr(activation_result, "policy_source", None)
    if not isinstance(fallback, bool):
        raise JITInvocationEvidenceInvalid("activation fallback must be boolean")
    if not isinstance(policy_source, str) or policy_source not in _ACTIVATION_SOURCES:
        raise JITInvocationEvidenceInvalid("activation policy source is not allowed")
    protocol_fields = _canonical_unordered(
        getattr(activation_result, "protocol_consumed_fields", ()),
        "protocol_consumed_fields",
    )
    runtime_fields = _canonical_unordered(
        getattr(activation_result, "runtime_enforced_capabilities", ()),
        "runtime_enforced_fields",
    )
    reason_codes = []
    raw_reasons = getattr(activation_result, "reason_codes", ())
    if not isinstance(raw_reasons, (list, tuple)):
        raise JITInvocationEvidenceInvalid("activation reason_codes must be a collection")
    for reason in raw_reasons:
        if not isinstance(reason, str) or not reason:
            raise JITInvocationEvidenceInvalid("activation reason code is invalid")
        base = reason.split(":", 1)[0]
        if base not in _ALLOWED_ACTIVATION_REASON_CODES:
            raise JITInvocationEvidenceInvalid(f"activation reason code is not allowed: {base}")
        reason_codes.append(base)
    return {
        "fallback": fallback,
        "active_policy": canonical_policy_serialization(active_policy),
        "policy_source": policy_source,
        "protocol_consumed_fields": list(protocol_fields or ()),
        "runtime_enforced_fields": list(runtime_fields or ()),
        "reason_codes": sorted(set(reason_codes)),
    }


def activation_semantic_hash(activation_result: object) -> Optional[str]:
    canonical = _canonical_activation_serialization(activation_result)
    return _sha256(canonical) if canonical is not None else None


def _safe_non_negative_int(value: object) -> Optional[int]:
    if value is None or (isinstance(value, str) and value in {"Unknown", "Unavailable"}):
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JITInvocationEvidenceInvalid("count must be a non-negative integer or null")
    return value


def _safe_non_negative_number(value: object) -> Optional[float]:
    if value is None or (isinstance(value, str) and value in {"Unknown", "Unavailable"}):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JITInvocationEvidenceInvalid("metric must be a non-negative number or null")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise JITInvocationEvidenceInvalid("metric must be finite and non-negative")
    return value


def _safe_hash(value: object, field: str) -> Optional[str]:
    if value is None or (isinstance(value, str) and value in {"Unknown", "Unavailable"}):
        return None
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise JITInvocationEvidenceInvalid(f"{field} must be a lowercase SHA-256 hash or null")
    return value


def _safe_status(value: object, field: str, allowed: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise JITInvocationEvidenceInvalid(f"{field} is not an allowed status")
    return value


def _safe_field_collection(value: object, field: str) -> Optional[Tuple[str, ...]]:
    normalized = _canonical_unordered(value, field)
    if normalized is not None:
        unknown = set(normalized) - set(_POLICY_FIELDS)
        if unknown:
            raise JITInvocationEvidenceInvalid(f"{field} contains unknown policy fields")
    return normalized


def _safe_policy_task_class(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and value in {"Unknown", "Unavailable"}:
        return None
    if not isinstance(value, str) or value not in set(TASK_CLASSES):
        raise JITInvocationEvidenceInvalid("policy_task_class is not an allowed task class")
    return value


def _prepared_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class JITInvocationEvidence:
    task_id: str
    evidence_version: str
    prepared_at: str
    policy_task_class: Optional[str]
    policy_valid: Optional[bool]
    policy_fallback: Optional[bool]
    policy_compile_ms: Optional[float]
    base_policy_hash: Optional[str]
    adapted_policy_hash: Optional[str]
    activation_result_hash: Optional[str]
    j3_status: str
    experience_advice_count: Optional[int]
    adapted_field_count: Optional[int]
    experience_conflict_count: Optional[int]
    experience_policy_sources: Optional[Tuple[str, ...]]
    j4_status: str
    policy_applied_fields: Optional[Tuple[str, ...]]
    runtime_enforced_fields: Optional[Tuple[str, ...]]
    protocol_consumed_fields: Optional[Tuple[str, ...]]
    activation_enforcement: str

    def as_mapping(self) -> Dict[str, object]:
        self.validate()
        policy_applied_fields = _safe_field_collection(
            self.policy_applied_fields, "policy_applied_fields"
        )
        runtime_enforced_fields = _safe_field_collection(
            self.runtime_enforced_fields, "runtime_enforced_fields"
        )
        protocol_consumed_fields = _safe_field_collection(
            self.protocol_consumed_fields, "protocol_consumed_fields"
        )
        return {
            "task_id": self.task_id,
            "evidence_version": self.evidence_version,
            "prepared_at": self.prepared_at,
            "policy_task_class": self.policy_task_class,
            "policy_valid": self.policy_valid,
            "policy_fallback": self.policy_fallback,
            "policy_compile_ms": self.policy_compile_ms,
            "base_policy_hash": self.base_policy_hash,
            "adapted_policy_hash": self.adapted_policy_hash,
            "activation_result_hash": self.activation_result_hash,
            "j3_status": self.j3_status,
            "experience_advice_count": self.experience_advice_count,
            "adapted_field_count": self.adapted_field_count,
            "experience_conflict_count": self.experience_conflict_count,
            "experience_policy_sources": list(self.experience_policy_sources)
            if self.experience_policy_sources is not None
            else None,
            "j4_status": self.j4_status,
            "policy_applied_fields": list(policy_applied_fields)
            if policy_applied_fields is not None
            else None,
            "runtime_enforced_fields": list(runtime_enforced_fields)
            if runtime_enforced_fields is not None
            else None,
            "protocol_consumed_fields": list(protocol_consumed_fields)
            if protocol_consumed_fields is not None
            else None,
            "activation_enforcement": self.activation_enforcement,
        }

    def validate(self, expected_task_id: Optional[str] = None) -> None:
        if not isinstance(self.task_id, str) or not _TASK_ID_RE.fullmatch(self.task_id):
            raise JITInvocationEvidenceInvalid("task_id must be a UUID v4")
        if expected_task_id is not None and self.task_id != expected_task_id:
            raise JITInvocationEvidenceInvalid("task_id does not match requested evidence")
        if self.evidence_version != JIT_INVOCATION_EVIDENCE_VERSION:
            raise JITInvocationEvidenceInvalid("unsupported evidence_version")
        if not isinstance(self.prepared_at, str) or not self.prepared_at:
            raise JITInvocationEvidenceInvalid("prepared_at is required")
        _safe_policy_task_class(self.policy_task_class)
        for name in ("policy_valid", "policy_fallback"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise JITInvocationEvidenceInvalid(f"{name} must be boolean or null")
        _safe_non_negative_number(self.policy_compile_ms)
        _safe_hash(self.base_policy_hash, "base_policy_hash")
        _safe_hash(self.adapted_policy_hash, "adapted_policy_hash")
        _safe_hash(self.activation_result_hash, "activation_result_hash")
        _safe_status(self.j3_status, "j3_status", _J3_STATUSES)
        _safe_status(self.j4_status, "j4_status", _J4_STATUSES)
        _safe_non_negative_int(self.experience_advice_count)
        _safe_non_negative_int(self.adapted_field_count)
        _safe_non_negative_int(self.experience_conflict_count)
        _canonical_ordered(self.experience_policy_sources, "experience_policy_sources")
        _safe_field_collection(self.policy_applied_fields, "policy_applied_fields")
        _safe_field_collection(self.runtime_enforced_fields, "runtime_enforced_fields")
        _safe_field_collection(self.protocol_consumed_fields, "protocol_consumed_fields")
        if self.activation_enforcement != "Protocol Only":
            raise JITInvocationEvidenceInvalid("activation_enforcement must be Protocol Only")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], expected_task_id: Optional[str] = None
    ) -> "JITInvocationEvidence":
        if not isinstance(value, Mapping):
            raise JITInvocationEvidenceInvalid("evidence must be a mapping")
        if set(value) != set(_EVIDENCE_FIELDS):
            raise JITInvocationEvidenceInvalid("evidence fields do not match the allowlist")
        evidence = cls(
            task_id=value["task_id"],
            evidence_version=value["evidence_version"],
            prepared_at=value["prepared_at"],
            policy_task_class=_safe_policy_task_class(value["policy_task_class"]),
            policy_valid=value["policy_valid"],
            policy_fallback=value["policy_fallback"],
            policy_compile_ms=_safe_non_negative_number(value["policy_compile_ms"]),
            base_policy_hash=_safe_hash(value["base_policy_hash"], "base_policy_hash"),
            adapted_policy_hash=_safe_hash(value["adapted_policy_hash"], "adapted_policy_hash"),
            activation_result_hash=_safe_hash(value["activation_result_hash"], "activation_result_hash"),
            j3_status=_safe_status(value["j3_status"], "j3_status", _J3_STATUSES),
            experience_advice_count=_safe_non_negative_int(value["experience_advice_count"]),
            adapted_field_count=_safe_non_negative_int(value["adapted_field_count"]),
            experience_conflict_count=_safe_non_negative_int(value["experience_conflict_count"]),
            experience_policy_sources=_canonical_ordered(
                value["experience_policy_sources"], "experience_policy_sources"
            ),
            j4_status=_safe_status(value["j4_status"], "j4_status", _J4_STATUSES),
            policy_applied_fields=_safe_field_collection(value["policy_applied_fields"], "policy_applied_fields"),
            runtime_enforced_fields=_safe_field_collection(
                value["runtime_enforced_fields"], "runtime_enforced_fields"
            ),
            protocol_consumed_fields=_safe_field_collection(
                value["protocol_consumed_fields"], "protocol_consumed_fields"
            ),
            activation_enforcement=value["activation_enforcement"],
        )
        evidence.validate(expected_task_id=expected_task_id)
        return evidence

    @classmethod
    def from_formal_task_result(
        cls, result: object, prepared_at: Optional[str] = None
    ) -> "JITInvocationEvidence":
        task_id = getattr(result, "task_id", None)
        if not isinstance(task_id, str):
            raise JITInvocationEvidenceInvalid("FormalTaskJITResult task_id is required")
        evidence = getattr(result, "evidence", {})
        if not isinstance(evidence, Mapping):
            raise JITInvocationEvidenceInvalid("FormalTaskJITResult evidence is invalid")
        j2 = evidence.get("j2") if isinstance(evidence.get("j2"), Mapping) else {}
        j3 = evidence.get("j3") if isinstance(evidence.get("j3"), Mapping) else {}
        j4 = evidence.get("j4") if isinstance(evidence.get("j4"), Mapping) else {}
        compiler = getattr(result, "compiler_result", None)
        adaptation = getattr(result, "adaptation_result", None)
        activation = getattr(result, "activation_result", None)
        policy = getattr(compiler, "policy", None)
        adapted_policy = getattr(adaptation, "policy", None)
        policy_valid = getattr(compiler, "valid", None)
        policy_fallback = getattr(compiler, "fallback", None)
        if policy_valid is not None and not isinstance(policy_valid, bool):
            policy_valid = None
        if policy_fallback is not None and not isinstance(policy_fallback, bool):
            policy_fallback = None
        source_ids = j3.get("experience_policy_sources")
        if isinstance(source_ids, str) and source_ids in {"Unknown", "Unavailable"}:
            source_ids = None
        evidence_value = cls(
            task_id=task_id,
            evidence_version=JIT_INVOCATION_EVIDENCE_VERSION,
            prepared_at=prepared_at or _prepared_at(),
            policy_task_class=_safe_policy_task_class(j2.get("task_class")),
            policy_valid=policy_valid,
            policy_fallback=policy_fallback,
            policy_compile_ms=_safe_non_negative_number(j2.get("policy_compile_ms")),
            base_policy_hash=policy_semantic_hash(policy),
            adapted_policy_hash=policy_semantic_hash(adapted_policy),
            activation_result_hash=activation_semantic_hash(activation),
            j3_status=_safe_status(j3.get("experience_adaptation", "Unknown"), "j3_status", _J3_STATUSES),
            experience_advice_count=_safe_non_negative_int(j3.get("eligible_experience_advice")),
            adapted_field_count=_safe_non_negative_int(
                len(j3.get("adapted_fields"))
                if isinstance(j3.get("adapted_fields"), list)
                else j3.get("adapted_fields")
            ),
            experience_conflict_count=_safe_non_negative_int(j3.get("experience_conflicts")),
            experience_policy_sources=_canonical_ordered(source_ids, "experience_policy_sources"),
            j4_status=_safe_status(j4.get("policy_activation", "Unknown"), "j4_status", _J4_STATUSES),
            policy_applied_fields=_safe_field_collection(
                None
                if isinstance(j4.get("applied_policy_fields"), str)
                and j4.get("applied_policy_fields") in {"Unknown", "Unavailable"}
                else j4.get("applied_policy_fields"),
                "policy_applied_fields",
            ),
            runtime_enforced_fields=_safe_field_collection(
                None
                if isinstance(j4.get("runtime_enforced_fields"), str)
                and j4.get("runtime_enforced_fields") in {"Unknown", "Unavailable"}
                else j4.get("runtime_enforced_fields"),
                "runtime_enforced_fields",
            ),
            protocol_consumed_fields=_safe_field_collection(
                None
                if isinstance(j4.get("protocol_consumed_fields"), str)
                and j4.get("protocol_consumed_fields") in {"Unknown", "Unavailable"}
                else j4.get("protocol_consumed_fields"),
                "protocol_consumed_fields",
            ),
            activation_enforcement=j4.get("policy_activation_enforcement", "Protocol Only"),
        )
        evidence_value.validate()
        return evidence_value

    def semantic_mapping(self) -> Dict[str, object]:
        mapping = self.as_mapping()
        return {field: mapping[field] for field in _SEMANTIC_EVIDENCE_FIELDS}


@dataclass(frozen=True)
class JITEvidencePersistResult:
    status: str
    evidence: Optional[JITInvocationEvidence]
    reason: Optional[str] = None


@dataclass(frozen=True)
class JITInvocationAudit:
    status: str
    source: str
    evidence: Optional[JITInvocationEvidence]
    reason: Optional[str] = None
    insufficient_sources: Tuple[str, ...] = ()


def jit_invocation_evidence_path(task_id: str, codex_home: Optional[Path] = None) -> Path:
    if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id):
        raise JITInvocationEvidenceInvalid("task_id must be a UUID v4")
    configured_home = codex_home or os.environ.get("CODEX_HOME")
    root = Path(configured_home).expanduser() if configured_home else Path.home() / ".codex"
    return root / ".state" / "experience-runtime" / "jit-invocation" / f"{task_id}.json"


def _read_existing(path: Path, task_id: str) -> JITInvocationEvidence:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise JITInvocationEvidenceInvalid("existing evidence is corrupt or unreadable") from exc
    return JITInvocationEvidence.from_mapping(payload, expected_task_id=task_id)


def _publish_no_replace(temp_path: Path, target: Path) -> None:
    """Publish a complete temp file without replacing an existing target."""

    try:
        os.link(str(temp_path), str(target))
    except FileExistsError:
        raise
    except OSError:
        if os.name != "nt":
            raise JITInvocationEvidenceError("filesystem lacks atomic no-replace publish")
        try:
            # On Windows, os.rename fails with FileExistsError instead of
            # replacing an existing destination.
            os.rename(str(temp_path), str(target))
        except FileExistsError:
            raise
        except OSError as exc:
            raise JITInvocationEvidenceError("Windows no-replace publish failed") from exc


def _candidate_evidence(
    value: Union[JITInvocationEvidence, object], prepared_at: Optional[str] = None
) -> JITInvocationEvidence:
    if isinstance(value, JITInvocationEvidence):
        value.validate()
        return value
    return JITInvocationEvidence.from_formal_task_result(value, prepared_at=prepared_at)


def persist_jit_invocation_evidence(
    value: Union[JITInvocationEvidence, object],
    *,
    codex_home: Optional[Path] = None,
    prepared_at: Optional[str] = None,
) -> JITEvidencePersistResult:
    """Create or reuse one task artifact; never overwrite a target."""

    candidate = _candidate_evidence(value, prepared_at=prepared_at)
    target = jit_invocation_evidence_path(candidate.task_id, codex_home=codex_home)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        try:
            existing = _read_existing(target, candidate.task_id)
        except JITInvocationEvidenceInvalid as exc:
            return JITEvidencePersistResult("conflict", None, "invalid_existing_evidence")
        if existing.semantic_mapping() == candidate.semantic_mapping():
            return JITEvidencePersistResult("reused", existing)
        return JITEvidencePersistResult("conflict", existing, "semantic_evidence_conflict")

    payload = _canonical_json(candidate.as_mapping())
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{candidate.task_id}.", suffix=".tmp", dir=str(target.parent)
    )
    temp_path = Path(temp_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            _publish_no_replace(temp_path, target)
            published = True
        except FileExistsError:
            existing = _read_existing(target, candidate.task_id)
            if existing.semantic_mapping() == candidate.semantic_mapping():
                return JITEvidencePersistResult("reused", existing)
            return JITEvidencePersistResult("conflict", existing, "semantic_evidence_conflict")
        read_back = _read_existing(target, candidate.task_id)
        if read_back.semantic_mapping() != candidate.semantic_mapping():
            raise JITInvocationEvidenceError("published evidence failed canonical read-back")
        return JITEvidencePersistResult("created", read_back)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                if published:
                    raise JITInvocationEvidenceError("temporary evidence cleanup failed")


def read_jit_invocation_evidence(
    task_id: str, *, codex_home: Optional[Path] = None
) -> Optional[JITInvocationEvidence]:
    path = jit_invocation_evidence_path(task_id, codex_home=codex_home)
    if not path.exists():
        return None
    return _read_existing(path, task_id)


def _metrics_mapping(metrics: object) -> Mapping[str, object]:
    if isinstance(metrics, Mapping):
        return metrics
    as_mapping = getattr(metrics, "as_mapping", None)
    if callable(as_mapping):
        value = as_mapping()
        if isinstance(value, Mapping):
            return value
    return {}


def _metrics_semantic_mapping(
    task_id: str, metrics: object
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    values = _metrics_mapping(metrics)
    if not values:
        return None, "metrics_unavailable"
    marker = values.get("evidence_version", values.get("jit_policy_evidence_version"))
    if marker != JIT_INVOCATION_EVIDENCE_VERSION:
        return None, "metrics_evidence_version_missing"
    required = set(_SEMANTIC_EVIDENCE_FIELDS)
    if not required.issubset(values):
        return None, "insufficient_for_semantic_comparison"
    try:
        raw = dict(values)
        raw["evidence_version"] = marker
        raw.setdefault("prepared_at", "metrics")
        raw.setdefault("policy_compile_ms", None)
        evidence = JITInvocationEvidence.from_mapping(raw, expected_task_id=task_id)
    except JITInvocationEvidenceInvalid:
        return None, "insufficient_for_semantic_comparison"
    return evidence.semantic_mapping(), None


def audit_jit_invocation_evidence(
    task_id: str,
    *,
    codex_home: Optional[Path] = None,
    current_result: Optional[object] = None,
    terminal_metrics: Optional[object] = None,
) -> JITInvocationAudit:
    """Resolve JIT evidence without rerunning J2, J3, or J4."""

    durable: Optional[JITInvocationEvidence]
    try:
        durable = read_jit_invocation_evidence(task_id, codex_home=codex_home)
    except JITInvocationEvidenceInvalid as exc:
        return JITInvocationAudit("JIT Evidence Conflict", "durable_artifact", None, str(exc))

    insufficient = []
    if durable is not None:
        durable_semantic = durable.semantic_mapping()
        if current_result is not None:
            try:
                current = JITInvocationEvidence.from_formal_task_result(current_result)
                if current.task_id != task_id:
                    return JITInvocationAudit("JIT Evidence Conflict", "durable_artifact", durable, "task_id_conflict")
                if current.semantic_mapping() != durable_semantic:
                    return JITInvocationAudit("JIT Evidence Conflict", "durable_artifact", durable, "current_result_conflict")
            except JITInvocationEvidenceError as exc:
                return JITInvocationAudit("JIT Evidence Conflict", "durable_artifact", durable, str(exc))
        if terminal_metrics is not None:
            metrics_semantic, reason = _metrics_semantic_mapping(task_id, terminal_metrics)
            if metrics_semantic is None:
                if reason == "insufficient_for_semantic_comparison":
                    insufficient.append("terminal_metrics")
            elif metrics_semantic != durable_semantic:
                return JITInvocationAudit("JIT Evidence Conflict", "durable_artifact", durable, "terminal_metrics_conflict")
        if durable.policy_fallback:
            return JITInvocationAudit("JIT Pipeline Fallback", "durable_artifact", durable, "policy_fallback", tuple(insufficient))
        return JITInvocationAudit("JIT Pipeline Executed", "durable_artifact", durable, None, tuple(insufficient))

    if current_result is not None:
        try:
            current = JITInvocationEvidence.from_formal_task_result(current_result)
            if current.task_id != task_id:
                return JITInvocationAudit("JIT Evidence Conflict", "current_result", None, "task_id_conflict")
            if current.policy_fallback:
                return JITInvocationAudit("JIT Pipeline Fallback", "current_result", current, "policy_fallback")
            return JITInvocationAudit("JIT Pipeline Executed", "current_result", current)
        except JITInvocationEvidenceError:
            return JITInvocationAudit("Unable To Prove", "current_result", None, "current_result_insufficient")

    if terminal_metrics is not None:
        metrics_semantic, reason = _metrics_semantic_mapping(task_id, terminal_metrics)
        if metrics_semantic is not None:
            return JITInvocationAudit("JIT Pipeline Executed", "terminal_metrics", None)
        if reason == "insufficient_for_semantic_comparison":
            insufficient.append("terminal_metrics")

    return JITInvocationAudit("Unable To Prove", "none", None, "evidence_unavailable", tuple(insufficient))


__all__ = [
    "JIT_INVOCATION_EVIDENCE_VERSION",
    "JITInvocationEvidence",
    "JITEvidencePersistResult",
    "JITInvocationAudit",
    "JITInvocationEvidenceError",
    "JITInvocationEvidenceConflict",
    "JITInvocationEvidenceInvalid",
    "canonical_policy_serialization",
    "policy_semantic_hash",
    "activation_semantic_hash",
    "jit_invocation_evidence_path",
    "persist_jit_invocation_evidence",
    "read_jit_invocation_evidence",
    "audit_jit_invocation_evidence",
]
