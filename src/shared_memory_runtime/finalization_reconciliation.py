"""Read-only reconciliation of local Finalization State and shared evidence."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .finalization_receipt import (
    TERMINAL_PAIRS,
    SharedFinalizationEvidence,
    inspect_shared_finalization_evidence,
    verify_finalization_receipt,
)
from .markdown import FragmentValidationError, project_key, project_key_aliases


RECOVERABLE_OUTSTANDING = "Recoverable Outstanding Finalization"
STALE_LOCAL_STATE = "Stale Local Finalization State"
EVIDENCE_CONFLICT = "Finalization Evidence Conflict"
UNRESOLVED_LOCAL_TASK = "Unresolved Local Formal Task"
TERMINAL = "Terminal"

_RECOVERY_EVIDENCE_RE = re.compile(
    r"^Recovery Evidence:\s*\n"
    r"Task Result:\s*(?P<task_result>Passed|Blocked|Undetermined)\s*\n"
    r"Primary:\s*(?P<primary>Found|Missing)\s*\n"
    r"Primary Hash:\s*(?P<primary_hash>[0-9a-f]{64}|None|Unknown)\s*\n"
    r"Missing Requirements:\s*(?P<missing>None|[^\n]+)\s*\n"
    r"Git Evidence:\s*Commit=(?P<commit>[0-9a-f]{7,64}|None);\s*"
    r"Push=(?P<push>Success|Failed|Unknown);\s*"
    r"Remote Verify=(?P<remote_verify>Success|Failed|Unknown);\s*"
    r"Reachable=(?P<reachable>Yes|No|Unknown)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class RecoveryEvidence:
    task_result: str
    primary: str
    primary_hash: Optional[str]
    missing_requirements: str
    commit: Optional[str]
    push: str
    remote_verify: str
    reachable: str


@dataclass(frozen=True)
class ReconciliationEntry:
    task_id: Optional[str]
    classification: str
    action: str
    shared_evidence_complete: bool
    local_terminal_receipt: str
    primary_path: Optional[str] = None
    primary_id: Optional[str] = None
    shared_commit: Optional[str] = None
    reason_codes: tuple[str, ...] = ()

    def as_mapping(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "classification": self.classification,
            "action": self.action,
            "shared_evidence_complete": self.shared_evidence_complete,
            "local_terminal_receipt": self.local_terminal_receipt,
            "primary_path": self.primary_path,
            "primary_id": self.primary_id,
            "shared_commit": self.shared_commit,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class ReconciliationReport:
    entries: tuple[ReconciliationEntry, ...]

    @property
    def recoverable_outstanding(self) -> tuple[str, ...]:
        return tuple(
            entry.task_id
            for entry in self.entries
            if entry.task_id and entry.classification == RECOVERABLE_OUTSTANDING
        )

    @property
    def stale_local_states(self) -> tuple[str, ...]:
        return tuple(
            entry.task_id
            for entry in self.entries
            if entry.task_id and entry.classification == STALE_LOCAL_STATE
        )

    @property
    def evidence_conflicts(self) -> tuple[str, ...]:
        return tuple(
            entry.task_id
            for entry in self.entries
            if entry.task_id and entry.classification == EVIDENCE_CONFLICT
        )

    @property
    def unresolved_local_tasks(self) -> tuple[str, ...]:
        return tuple(
            entry.task_id
            for entry in self.entries
            if entry.task_id and entry.classification == UNRESOLVED_LOCAL_TASK
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "recoverable_outstanding_finalization": list(self.recoverable_outstanding),
            "stale_local_finalization_state": list(self.stale_local_states),
            "finalization_evidence_conflict": list(self.evidence_conflicts),
            "unresolved_local_formal_task": list(self.unresolved_local_tasks),
            "entries": [entry.as_mapping() for entry in self.entries],
        }


def parse_recovery_evidence(validation_summary: str) -> Optional[RecoveryEvidence]:
    """Parse only the fixed Recovery Evidence block; prose is not accepted."""

    match = _RECOVERY_EVIDENCE_RE.search(validation_summary or "")
    if not match:
        return None
    values = match.groupdict()
    return RecoveryEvidence(
        task_result=values["task_result"],
        primary=values["primary"],
        primary_hash=(
            values["primary_hash"]
            if values["primary_hash"] not in {"None", "Unknown"}
            else None
        ),
        missing_requirements=values["missing"],
        commit=values["commit"] if values["commit"] != "None" else None,
        push=values["push"],
        remote_verify=values["remote_verify"],
        reachable=values["reachable"],
    )


def _state_ownership_reasons(state_path: Path, evidence: SharedFinalizationEvidence) -> list[str]:
    state = evidence.state
    if state is None:
        return []
    reasons: list[str] = []
    required = {
        "project_identity": state.get("project_identity"),
        "project_key": evidence.project_key,
        "project_environment": state.get("project_environment"),
        "worktree_root": state.get("worktree_root"),
        "worktree_key": state.get("worktree_key"),
    }
    if not all(isinstance(value, str) and value.strip() for value in required.values()):
        return ["state_ownership_unproven"]

    expected_name = (
        f"{evidence.project_key}--{state['worktree_key']}--{evidence.task_id}.json"
    )
    if state_path.name != expected_name:
        reasons.append("state_filename_identity_conflict")

    try:
        identity_key = project_key(str(state["project_identity"]))
        aliases = set(project_key_aliases(str(state["project_identity"])))
    except FragmentValidationError:
        reasons.append("invalid_project_identity")
    else:
        if evidence.project_key not in {identity_key, *aliases}:
            reasons.append("project_key_identity_conflict")

    worktree_value = str(state["project_environment"]) + "\0" + str(
        Path(str(state["worktree_root"])).expanduser().resolve()
    )
    expected_worktree_key = hashlib.sha256(worktree_value.encode("utf-8")).hexdigest()[:12]
    if state["worktree_key"] != expected_worktree_key:
        reasons.append("worktree_identity_conflict")
    return reasons


def _summary(state: Optional[dict[str, object]]) -> Optional[str]:
    if not state:
        return None
    value = state.get("validation_summary")
    return value if isinstance(value, str) else None


def _entry(
    task_id: Optional[str],
    classification: str,
    action: str,
    *,
    shared_complete: bool = False,
    local_receipt: str = "not_proven",
    evidence: Optional[SharedFinalizationEvidence] = None,
    reasons: Iterable[str] = (),
) -> ReconciliationEntry:
    return ReconciliationEntry(
        task_id=task_id,
        classification=classification,
        action=action,
        shared_evidence_complete=shared_complete,
        local_terminal_receipt=local_receipt,
        primary_path=evidence.primary_path if evidence else None,
        primary_id=evidence.primary_id if evidence else None,
        shared_commit=evidence.shared_commit if evidence else None,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def reconcile_finalization_state(
    source_root: Path,
    state_path: Path,
    *,
    reported_task_id: Optional[str] = None,
) -> ReconciliationEntry:
    """Classify one local State without writing State or creating a Receipt."""

    evidence = inspect_shared_finalization_evidence(source_root, state_path)
    task_id = reported_task_id or evidence.task_id
    if evidence.state is None:
        return _entry(task_id, UNRESOLVED_LOCAL_TASK, "PRESERVE", reasons=evidence.reasons)

    ownership_reasons = _state_ownership_reasons(state_path, evidence)
    if ownership_reasons:
        return _entry(
            task_id,
            EVIDENCE_CONFLICT,
            "BLOCK",
            evidence=evidence,
            reasons=(*evidence.reasons, *ownership_reasons),
        )

    terminal_pair = (evidence.finalization_state, evidence.learning_stage)
    if terminal_pair in TERMINAL_PAIRS:
        receipt = verify_finalization_receipt(source_root, state_path)
        if receipt.complete:
            return _entry(
                task_id,
                TERMINAL,
                "CONTINUE",
                shared_complete=True,
                local_receipt="proven",
                evidence=evidence,
                reasons=receipt.reason_codes,
            )
        return _entry(
            task_id,
            EVIDENCE_CONFLICT,
            "BLOCK",
            evidence=evidence,
            reasons=receipt.reason_codes,
        )

    recovery = parse_recovery_evidence(_summary(evidence.state) or "")
    if recovery is None or recovery.task_result == "Undetermined":
        return _entry(
            task_id,
            UNRESOLVED_LOCAL_TASK,
            "PRESERVE",
            evidence=evidence,
            reasons=(*evidence.reasons, "recovery_evidence_undetermined"),
        )

    inferred_result = evidence.task_result
    if inferred_result and inferred_result != recovery.task_result:
        return _entry(
            task_id,
            EVIDENCE_CONFLICT,
            "BLOCK",
            evidence=evidence,
            reasons=(*evidence.reasons, "task_result_conflict"),
        )

    if evidence.reasons and any(reason.startswith("state_") for reason in evidence.reasons):
        return _entry(
            task_id,
            UNRESOLVED_LOCAL_TASK,
            "PRESERVE",
            evidence=evidence,
            reasons=(*evidence.reasons, "state_evidence_incomplete"),
        )

    if recovery.primary == "Missing":
        if evidence.primary_path:
            return _entry(
                task_id,
                EVIDENCE_CONFLICT,
                "BLOCK",
                evidence=evidence,
                reasons=(*evidence.reasons, "primary_presence_conflict"),
            )
        return _entry(
            task_id,
            RECOVERABLE_OUTSTANDING,
            "RESUME",
            evidence=evidence,
            reasons=(*evidence.reasons, "primary_missing"),
        )

    if not evidence.primary_path or not evidence.primary_id:
        return _entry(
            task_id,
            EVIDENCE_CONFLICT if evidence.reasons else RECOVERABLE_OUTSTANDING,
            "BLOCK" if evidence.reasons else "RESUME",
            evidence=evidence,
            reasons=(*evidence.reasons, "primary_unavailable"),
        )

    if recovery.primary_hash is None:
        return _entry(
            task_id,
            RECOVERABLE_OUTSTANDING,
            "RESUME",
            evidence=evidence,
            reasons=(*evidence.reasons, "primary_hash_unproven"),
        )
    if recovery.primary_hash != evidence.primary_hash:
        return _entry(
            task_id,
            EVIDENCE_CONFLICT,
            "BLOCK",
            evidence=evidence,
            reasons=(*evidence.reasons, "primary_hash_conflict"),
        )

    if evidence.primary_status not in {"completed", "blocked"}:
        return _entry(
            task_id,
            EVIDENCE_CONFLICT,
            "BLOCK",
            evidence=evidence,
            reasons=(*evidence.reasons, "primary_terminal_status_unproven"),
        )
    if evidence.experience_verification != "verified":
        return _entry(
            task_id,
            EVIDENCE_CONFLICT,
            "BLOCK",
            evidence=evidence,
            reasons=(*evidence.reasons, "experience_verification_unproven"),
        )

    if "remote_primary_hash_conflict" in evidence.reasons:
        return _entry(
            task_id,
            EVIDENCE_CONFLICT,
            "BLOCK",
            evidence=evidence,
            reasons=evidence.reasons,
        )

    if not evidence.proof_verified:
        return _entry(
            task_id,
            RECOVERABLE_OUTSTANDING,
            "RESUME",
            evidence=evidence,
            reasons=evidence.reasons,
        )

    return _entry(
        task_id,
        STALE_LOCAL_STATE,
        "CONTINUE",
        shared_complete=True,
        local_receipt="not_proven",
        evidence=evidence,
        reasons=(*evidence.reasons, "shared_evidence_complete_local_receipt_not_proven"),
    )


def _candidate_paths(
    state_root: Path,
    state_paths: Sequence[Path],
    task_ids: Sequence[str],
) -> list[tuple[Path, Optional[str]]]:
    if state_paths or task_ids:
        candidates: list[tuple[Path, Optional[str]]] = []
        for state_path in state_paths:
            candidates.append((Path(state_path), None))
        for task_id in task_ids:
            matches = sorted(state_root.glob(f"*--{task_id}.json"))
            if len(matches) == 1:
                candidates.append((matches[0], task_id))
            elif not matches:
                candidates.append((state_root / f"missing--{task_id}.json", task_id))
            else:
                candidates.extend((match, task_id) for match in matches)
        return candidates
    return [(path, None) for path in sorted(state_root.glob("*.json"))]


def reconcile_finalization_states(
    source_root: Path,
    state_root: Path,
    *,
    state_paths: Sequence[Path] = (),
    task_ids: Sequence[str] = (),
) -> ReconciliationReport:
    """Reconcile only supplied/local State files; never scan historical Fragments."""

    entries = tuple(
        reconcile_finalization_state(source_root, path, reported_task_id=task_id)
        for path, task_id in _candidate_paths(Path(state_root), state_paths, task_ids)
    )
    return ReconciliationReport(entries=entries)


__all__ = [
    "EVIDENCE_CONFLICT",
    "RECOVERABLE_OUTSTANDING",
    "STALE_LOCAL_STATE",
    "TERMINAL",
    "UNRESOLVED_LOCAL_TASK",
    "RecoveryEvidence",
    "ReconciliationEntry",
    "ReconciliationReport",
    "parse_recovery_evidence",
    "reconcile_finalization_state",
    "reconcile_finalization_states",
]
