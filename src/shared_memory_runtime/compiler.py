"""One deterministic Terminal Experience Compiler operation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from .models import Applicability, ExperienceRecord


@dataclass(frozen=True)
class DraftExperience:
    experience_id: str
    scope: str
    project_key: Optional[str]
    kind: str
    trigger: str
    core_action: str
    applicability: Applicability
    does_not_apply: str
    exceptions: str
    anchors: Tuple[str, ...]
    correction_of: Optional[str] = None
    reuse_proven: bool = False


@dataclass(frozen=True)
class SemanticCandidate:
    experience_id: str
    canonical_id: str
    scope: str
    project_key: Optional[str]
    kind: str
    trigger: str
    core_action: str
    applicability: Applicability
    does_not_apply: str
    exceptions: str
    anchors: Tuple[str, ...]

    @classmethod
    def from_record(cls, record: ExperienceRecord) -> "SemanticCandidate":
        return cls(
            experience_id=record.id,
            canonical_id=record.canonical_id,
            scope=record.scope,
            project_key=record.project_key,
            kind=record.kind,
            trigger=record.trigger,
            core_action=record.core_action,
            applicability=record.applicability,
            does_not_apply=record.does_not_apply,
            exceptions=record.exceptions,
            anchors=record.anchor_values(),
        )


@dataclass(frozen=True)
class CompilerResult:
    outcome: str
    canonical_id: str
    candidate_ids: Tuple[str, ...]
    model_required: bool
    reason_code: Optional[str]


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _equivalent(draft: DraftExperience, candidate: SemanticCandidate) -> bool:
    return (
        _normalize(draft.trigger) == _normalize(candidate.trigger)
        and _normalize(draft.core_action) == _normalize(candidate.core_action)
        and draft.applicability.semantic_key() == candidate.applicability.semantic_key()
        and _normalize(draft.does_not_apply) == _normalize(candidate.does_not_apply)
        and _normalize(draft.exceptions) == _normalize(candidate.exceptions)
        and draft.scope == candidate.scope
        and draft.project_key == candidate.project_key
        and draft.kind == candidate.kind
    )


def _semantics_complete(draft: DraftExperience, candidate: SemanticCandidate) -> bool:
    return all(
        (
            draft.trigger,
            draft.core_action,
            candidate.trigger,
            candidate.core_action,
        )
    )


def compile_terminal_experience(
    draft: DraftExperience,
    candidates: Sequence[Union[SemanticCandidate, ExperienceRecord]],
) -> CompilerResult:
    """Return NEW/REINFORCE/CORRECT without a default model call.

    `model_required` is an explicit escalation record, not an invocation. The
    caller may send one minimal semantic judgment only when the returned
    reason code permits it.
    """

    semantic_candidates = [
        candidate if isinstance(candidate, SemanticCandidate) else SemanticCandidate.from_record(candidate)
        for candidate in candidates
    ]
    if draft.correction_of:
        return CompilerResult("CORRECT", draft.experience_id, (draft.correction_of,), False, None)

    applicability = draft.applicability
    if set(applicability.project_keys) & set(applicability.excluded_project_keys):
        return CompilerResult(
            "NEW",
            draft.experience_id,
            (),
            True,
            "complex_applicability_ambiguity",
        )
    if set(applicability.task_kinds) & set(applicability.excluded_task_kinds):
        return CompilerResult(
            "NEW",
            draft.experience_id,
            (),
            True,
            "complex_applicability_ambiguity",
        )

    draft_anchors = {value.casefold() for value in draft.anchors}
    exact = [
        candidate
        for candidate in semantic_candidates
        if draft_anchors & {value.casefold() for value in candidate.anchors}
    ]
    if not exact:
        return CompilerResult("NEW", draft.experience_id, (), False, None)

    if not _semantics_complete(draft, exact[0]):
        return CompilerResult(
            "NEW",
            draft.experience_id,
            tuple(candidate.experience_id for candidate in exact),
            True,
            "new_correct_ambiguity",
        )

    equivalent = [candidate for candidate in exact if _equivalent(draft, candidate)]
    canonical_ids = {candidate.canonical_id for candidate in equivalent}
    if len(canonical_ids) > 1:
        return CompilerResult(
            "NEW",
            draft.experience_id,
            tuple(candidate.experience_id for candidate in equivalent),
            True,
            "semantic_conflict",
        )
    if equivalent:
        canonical_id = next(iter(canonical_ids))
        if draft.reuse_proven:
            return CompilerResult(
                "REINFORCE",
                canonical_id,
                tuple(candidate.experience_id for candidate in equivalent),
                False,
                None,
            )
        return CompilerResult(
            "NEW",
            draft.experience_id,
            tuple(candidate.experience_id for candidate in equivalent),
            True,
            "new_correct_ambiguity",
        )
    return CompilerResult(
        "NEW",
        draft.experience_id,
        tuple(candidate.experience_id for candidate in exact),
        False,
        None,
    )
