"""Read-only terminal evidence verification for one formal Task."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .git_proof import GitEvidence
from .markdown import (
    FORMAL_PRIMARY_TYPES,
    FragmentValidationError,
    parse_experience_fragment,
    parse_front_matter,
)


TERMINAL_PAIRS = {
    ("sedimented", "remote_verified"),
    ("blocked", "remote_verified"),
}


@dataclass(frozen=True)
class FinalizationReceipt:
    """The evidence required before a formal Task may be reported terminal."""

    task_id: Optional[str]
    task_result: Optional[str]
    experience_outcome: Optional[str]
    primary_path: Optional[str]
    primary_id: Optional[str]
    shared_commit: Optional[str]
    push_verified: bool
    remote_verified: bool
    finalization_state: Optional[str]
    learning_stage: Optional[str]
    complete: bool
    reason_codes: tuple[str, ...]

    def as_mapping(self) -> dict[str, Any]:
        result = asdict(self)
        result["reason_codes"] = list(self.reason_codes)
        return result


@dataclass(frozen=True)
class SharedFinalizationEvidence:
    """Shared-side evidence inspected without claiming a local Receipt."""

    state: Optional[Mapping[str, Any]]
    task_id: Optional[str]
    project_key: Optional[str]
    finalization_state: Optional[str]
    learning_stage: Optional[str]
    task_result: Optional[str]
    experience_outcome: Optional[str]
    primary_path: Optional[str]
    primary_id: Optional[str]
    primary_type: Optional[str]
    primary_status: Optional[str]
    experience_verification: Optional[str]
    primary_hash: Optional[str]
    shared_commit: Optional[str]
    push_verified: bool
    remote_primary_hash: Optional[str]
    reasons: tuple[str, ...]

    @property
    def proof_verified(self) -> bool:
        return self.push_verified and self.shared_commit is not None


def _read_state(path: Path) -> tuple[Optional[Mapping[str, Any]], list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, ["state_missing"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, ["state_unreadable"]
    if not isinstance(payload, Mapping):
        return None, ["state_not_mapping"]
    return payload, []


def _state_string(state: Mapping[str, Any], key: str, reasons: list[str]) -> Optional[str]:
    value = state.get(key)
    if not isinstance(value, str) or not value.strip():
        reasons.append(f"state_{key}_missing")
        return None
    return value.strip()


def _primary_candidates(source_root: Path, project_key: str, task_id: str) -> list[Path]:
    primary_root = source_root / ".memory" / "projects" / project_key
    candidates: list[Path] = []
    for path in sorted(primary_root.glob("*/*.md")):
        try:
            metadata, _ = parse_front_matter(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, FragmentValidationError):
            continue
        if metadata.get("type") not in FORMAL_PRIMARY_TYPES:
            continue
        if str(metadata.get("task_id") or "").strip() == task_id:
            candidates.append(path)
    return candidates


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_result(finalization_state: Optional[str], learning_stage: Optional[str]) -> Optional[str]:
    if finalization_state == "blocked":
        return "Blocked"
    if finalization_state in {"verified_pending", "sedimented"}:
        return "Passed"
    return None


def inspect_shared_finalization_evidence(
    source_root: Path, state_path: Path
) -> SharedFinalizationEvidence:
    """Inspect shared-side evidence without requiring a local terminal pair."""

    reasons: list[str] = []
    state, state_reasons = _read_state(Path(state_path))
    reasons.extend(state_reasons)
    if state is None:
        return SharedFinalizationEvidence(
            state=None,
            task_id=None,
            project_key=None,
            finalization_state=None,
            learning_stage=None,
            task_result=None,
            experience_outcome=None,
            primary_path=None,
            primary_id=None,
            primary_type=None,
            primary_status=None,
            experience_verification=None,
            primary_hash=None,
            shared_commit=None,
            push_verified=False,
            remote_primary_hash=None,
            reasons=tuple(reasons),
        )

    task_id = _state_string(state, "task_id", reasons)
    project_key = _state_string(state, "project_key", reasons)
    finalization_state = _state_string(state, "finalization_state", reasons)
    learning_stage = _state_string(state, "learning_stage", reasons)
    task_result = _task_result(finalization_state, learning_stage)

    primary_path: Optional[str] = None
    primary_id: Optional[str] = None
    primary_type: Optional[str] = None
    primary_status: Optional[str] = None
    experience_verification: Optional[str] = None
    experience_outcome: Optional[str] = None
    primary_hash: Optional[str] = None
    shared_commit: Optional[str] = None
    push_verified = False
    remote_primary_hash: Optional[str] = None

    if task_id and project_key:
        candidates = _primary_candidates(Path(source_root), project_key, task_id)
        if not candidates:
            reasons.append("primary_missing")
        elif len(candidates) != 1:
            reasons.append("primary_not_exactly_one")
        else:
            primary = candidates[0]
            primary_path = primary.relative_to(Path(source_root)).as_posix()
            try:
                metadata, _ = parse_front_matter(primary.read_text(encoding="utf-8"))
                primary_type = str(metadata.get("type") or "").strip() or None
                primary_status = str(metadata.get("status") or "").strip() or None
                experience = metadata.get("experience")
                if isinstance(experience, Mapping):
                    value = experience.get("verification")
                    experience_verification = str(value).strip() if value else None
                primary_hash = _source_hash(primary)
                record = parse_experience_fragment(primary, Path(source_root), primary_hash)
            except (OSError, UnicodeDecodeError, FragmentValidationError) as exc:
                if isinstance(exc, FragmentValidationError) and exc.reason_code:
                    reasons.append(exc.reason_code)
                else:
                    reasons.append("primary_invalid")
            else:
                primary_id = record.id
                experience_outcome = record.outcome
                proof = GitEvidence(Path(source_root)).prove_remote_persistence(primary_path)
                shared_commit = proof.containing_revision
                push_verified = proof.verified
                if proof.remote_head:
                    remote_primary_hash = GitEvidence(Path(source_root)).blob_at(
                        proof.remote_head, primary_path
                    )
                if proof.source_hash and remote_primary_hash and proof.source_hash != remote_primary_hash:
                    reasons.append("remote_primary_hash_conflict")
                if not proof.verified:
                    reasons.append(f"remote_proof_{proof.reason_code}")

    return SharedFinalizationEvidence(
        state=state,
        task_id=task_id,
        project_key=project_key,
        finalization_state=finalization_state,
        learning_stage=learning_stage,
        task_result=task_result,
        experience_outcome=experience_outcome,
        primary_path=primary_path,
        primary_id=primary_id,
        primary_type=primary_type,
        primary_status=primary_status,
        experience_verification=experience_verification,
        primary_hash=primary_hash,
        shared_commit=shared_commit,
        push_verified=push_verified,
        remote_primary_hash=remote_primary_hash,
        reasons=tuple(reasons),
    )


def verify_finalization_receipt(source_root: Path, state_path: Path) -> FinalizationReceipt:
    """Verify terminal State, exactly-one Primary, and remote Git evidence.

    This function deliberately writes neither the Runtime State nor the source
    repository. It is safe to invoke from the Final Summary gate and returns a
    complete receipt only for the two legal terminal State/stage pairs.
    """

    evidence = inspect_shared_finalization_evidence(source_root, state_path)
    if evidence.state is None:
        return FinalizationReceipt(
            task_id=None,
            task_result=None,
            experience_outcome=None,
            primary_path=None,
            primary_id=None,
            shared_commit=None,
            push_verified=False,
            remote_verified=False,
            finalization_state=None,
            learning_stage=None,
            complete=False,
            reason_codes=evidence.reasons,
        )

    reasons = list(evidence.reasons)
    terminal_pair = (evidence.finalization_state, evidence.learning_stage)
    if terminal_pair not in TERMINAL_PAIRS:
        reasons.insert(0, "terminal_state_not_complete")

    remote_verified = evidence.proof_verified and evidence.learning_stage == "remote_verified"
    if evidence.learning_stage == "remote_verified" and not evidence.proof_verified:
        reasons.append("remote_verified_without_primary_remote_proof")

    complete = not reasons and terminal_pair in {
        ("sedimented", "remote_verified"),
        ("blocked", "remote_verified"),
    } and remote_verified
    if complete:
        reasons.append("receipt_complete")

    return FinalizationReceipt(
        task_id=evidence.task_id,
        task_result=evidence.task_result,
        experience_outcome=evidence.experience_outcome,
        primary_path=evidence.primary_path,
        primary_id=evidence.primary_id,
        shared_commit=evidence.shared_commit,
        push_verified=evidence.push_verified,
        remote_verified=remote_verified,
        finalization_state=evidence.finalization_state,
        learning_stage=evidence.learning_stage,
        complete=complete,
        reason_codes=tuple(reasons),
    )
