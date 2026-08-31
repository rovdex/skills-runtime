"""Bounded Experience Recall from the derived SQLite projection."""

from __future__ import annotations

import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .models import Applicability


@dataclass(frozen=True)
class RecallContext:
    project_key: Optional[str]
    task_kind: Optional[str] = None
    anchors: Tuple[str, ...] = ()
    query: str = ""
    task_class: str = "normal"
    expand: bool = False

    @property
    def limit(self) -> int:
        if self.task_class in {"complex", "debugging"}:
            return 6 if self.expand else 5
        return 4


@dataclass(frozen=True)
class RecallCandidate:
    experience_id: str
    canonical_id: str
    capsule: str
    score: float
    effective_experience_verification: str
    remote_verified: bool
    exact_anchor_hits: int
    applicability_passed: bool = True


@dataclass(frozen=True)
class RecallStats:
    query_ms: float
    candidate_count: int
    capsule_count: int
    approx_memory_tokens: int
    full_markdown_expansions: int
    exact_anchor_candidates: int = 0
    fts_candidates: int = 0
    trigram_candidates: int = 0
    fallback_candidates: int = 0


@dataclass(frozen=True)
class RecallRun:
    candidates: Tuple[RecallCandidate, ...]
    stats: RecallStats

    @classmethod
    def empty(cls) -> "RecallRun":
        return cls(
            candidates=(),
            stats=RecallStats(
                query_ms=0.0,
                candidate_count=0,
                capsule_count=0,
                approx_memory_tokens=0,
                full_markdown_expansions=0,
            ),
        )


@dataclass(frozen=True)
class _StoredExperience:
    experience_id: str
    canonical_id: str
    scope: str
    project_key: Optional[str]
    kind: str
    outcome: str
    fragment_status: str
    experience_verification: str
    title: str
    summary: str
    capsule: str
    confidence: str
    importance: int
    remote_verified: bool
    created_at: str
    trigger: str
    core_action: str
    does_not_apply: str
    exceptions: str
    anchors: Tuple[Tuple[str, str], ...]
    applicability: Applicability
    supersedes: Tuple[str, ...]
    reuse_count: int
    success_count: int
    failure_count: int
    last_used_at: Optional[str]
    last_result: Optional[str]

    @property
    def searchable_text(self) -> str:
        return " ".join(
            value
            for value in (
                self.title,
                self.summary,
                self.capsule,
                self.trigger,
                self.core_action,
                self.does_not_apply,
                self.exceptions,
            )
            if value
        ).casefold()


def _applicability_from_anchors(anchors: Sequence[Tuple[str, str]]) -> Applicability:
    values = defaultdict(list)
    for anchor_type, value in anchors:
        values[anchor_type].append(value)
    return Applicability(
        project_keys=tuple(sorted(values["applies.project_key"])),
        task_kinds=tuple(sorted(values["applies.task_kind"])),
        required_anchors=tuple(sorted(values["applies.required_anchor"])),
        excluded_project_keys=tuple(sorted(values["does_not_apply.project_key"])),
        excluded_task_kinds=tuple(sorted(values["does_not_apply.task_kind"])),
        excluded_anchors=tuple(sorted(values["does_not_apply.anchor"])),
    )


def _load_experiences(connection: sqlite3.Connection) -> List[_StoredExperience]:
    rows = connection.execute(
        """
        SELECT e.*, COALESCE(f.reuse_count, 0) AS reuse_count,
               COALESCE(f.success_count, 0) AS success_count,
               COALESCE(f.failure_count, 0) AS failure_count,
               f.last_used_at, f.last_result,
               COALESCE(t.trigger, '') AS trigger,
               COALESCE(t.core_action, '') AS core_action,
               COALESCE(t.does_not_apply, '') AS does_not_apply,
               COALESCE(t.exceptions, '') AS exceptions
        FROM experiences AS e
        LEFT JOIN experience_feedback AS f ON f.experience_id = e.id
        LEFT JOIN experience_fts AS t ON t.experience_id = e.id
        """
    ).fetchall()
    result = []
    for row in rows:
        anchor_rows = connection.execute(
            "SELECT anchor_type, anchor_value FROM experience_anchors WHERE experience_id = ?",
            (row["id"],),
        ).fetchall()
        anchors = tuple((item["anchor_type"], item["anchor_value"]) for item in anchor_rows)
        identity_values = defaultdict(list)
        for anchor_type, value in anchors:
            identity_values[anchor_type].append(value)
        canonical_id = identity_values["identity.canonical_id"]
        outcome = identity_values["identity.outcome"]
        if not canonical_id or not outcome:
            continue
        supersedes = tuple(value for anchor_type, value in anchors if anchor_type == "relation.supersedes")
        result.append(
            _StoredExperience(
                experience_id=row["id"],
                canonical_id=canonical_id[0],
                scope=row["scope"],
                project_key=row["project_key"],
                kind=row["kind"],
                outcome=outcome[0],
                fragment_status=row["fragment_status"],
                experience_verification=row["experience_verification"],
                title=row["title"],
                summary=row["summary"],
                capsule=row["capsule"],
                confidence=row["confidence"],
                importance=row["importance"],
                remote_verified=bool(row["remote_verified"]),
                created_at=row["created_at"],
                trigger=row["trigger"],
                core_action=row["core_action"],
                does_not_apply=row["does_not_apply"],
                exceptions=row["exceptions"],
                anchors=anchors,
                applicability=_applicability_from_anchors(anchors),
                supersedes=supersedes,
                reuse_count=row["reuse_count"],
                success_count=row["success_count"],
                failure_count=row["failure_count"],
                last_used_at=row["last_used_at"],
                last_result=row["last_result"],
            )
        )
    return result


def _scope_candidates(rows: Iterable[_StoredExperience], project_key: Optional[str]) -> List[_StoredExperience]:
    return [
        row
        for row in rows
        if (row.scope == "global" and row.project_key is None)
        or (row.scope == "project" and project_key is not None and row.project_key == project_key)
    ]


def _effective_verifications(rows: Sequence[_StoredExperience]) -> Mapping[str, str]:
    superseders: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        if row.experience_verification == "verified" and row.outcome == "CORRECT" and row.supersedes:
            for old_id in row.supersedes:
                superseders[old_id].add(row.experience_id)

    cache: Dict[str, str] = {}

    def effective(experience_id: str, trail: Set[str]) -> str:
        if experience_id in cache:
            return cache[experience_id]
        row = next(item for item in rows if item.experience_id == experience_id)
        if row.experience_verification != "verified":
            cache[experience_id] = row.experience_verification
            return row.experience_verification
        if experience_id in trail:
            cache[experience_id] = "invalid"
            return "invalid"
        if any(
            superseder not in trail and effective(superseder, trail | {experience_id}) == "verified"
            for superseder in superseders.get(experience_id, ())
        ):
            cache[experience_id] = "superseded"
        else:
            cache[experience_id] = "verified"
        return cache[experience_id]

    for row in rows:
        effective(row.experience_id, set())
    return cache


def _applicable(row: _StoredExperience, context: RecallContext) -> bool:
    applicability = row.applicability
    anchors = {value.casefold() for value in context.anchors}
    if applicability.project_keys and context.project_key not in applicability.project_keys:
        return False
    if context.project_key in applicability.excluded_project_keys:
        return False
    if applicability.task_kinds and context.task_kind not in applicability.task_kinds:
        return False
    if context.task_kind in applicability.excluded_task_kinds:
        return False
    if any(required.casefold() not in anchors for required in applicability.required_anchors):
        return False
    if any(excluded.casefold() in anchors for excluded in applicability.excluded_anchors):
        return False
    return True


@dataclass(frozen=True)
class _SearchScores:
    scores: Mapping[str, float]
    fts_ids: Set[str]
    trigram_ids: Set[str]


def _fts_scores(connection: sqlite3.Connection, query: str) -> _SearchScores:
    if not query.strip():
        return _SearchScores({}, set(), set())
    scores: Dict[str, float] = {}
    fts_ids: Set[str] = set()
    trigram_ids: Set[str] = set()
    terms = [term for term in re.split(r"\s+", query.strip()) if term]
    fts_query = " OR ".join(f'"{term.replace(chr(34), " ")}"' for term in terms)
    try:
        for row in connection.execute(
            "SELECT experience_id, bm25(experience_fts) AS rank FROM experience_fts WHERE experience_fts MATCH ?",
            (fts_query,),
        ):
            scores[row["experience_id"]] = max(scores.get(row["experience_id"], 0.0), -float(row["rank"]))
            fts_ids.add(row["experience_id"])
    except sqlite3.OperationalError:
        pass
    if len(query.strip()) >= 3:
        try:
            for row in connection.execute(
                "SELECT experience_id FROM experience_fts_trigram WHERE experience_fts_trigram MATCH ?",
                (query.strip(),),
            ):
                scores[row["experience_id"]] = max(scores.get(row["experience_id"], 0.0), 1.0)
                trigram_ids.add(row["experience_id"])
        except sqlite3.OperationalError:
            pass
    return _SearchScores(scores, fts_ids, trigram_ids)


def estimate_capsule_tokens(capsule: str) -> int:
    """Estimate injected tokens without adding a tokenizer dependency."""

    cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    tokens = 0
    ascii_run = []

    def flush_ascii() -> None:
        nonlocal tokens
        if ascii_run:
            tokens += (len(ascii_run) + 3) // 4
            ascii_run.clear()

    for character in capsule:
        if cjk.match(character):
            flush_ascii()
            tokens += 1
        elif character.isascii() and character.isalnum():
            ascii_run.append(character)
        else:
            flush_ascii()
            if not character.isspace():
                tokens += 1
    flush_ascii()
    return max(tokens, 1) if capsule.strip() else 0


def recall_with_stats(connection: sqlite3.Connection, context: RecallContext) -> RecallRun:
    """Recall only effective, remote-verified, applicable Capsules with metrics."""

    started_ns = time.perf_counter_ns()
    rows = _scope_candidates(_load_experiences(connection), context.project_key)
    if not rows:
        return RecallRun(
            candidates=(),
            stats=RecallStats(
                query_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
                candidate_count=0,
                capsule_count=0,
                approx_memory_tokens=0,
                full_markdown_expansions=0,
            ),
        )
    search_scores = _fts_scores(connection, context.query)
    context_anchor_values = {value.casefold() for value in context.anchors}
    effective = _effective_verifications(rows)
    candidates: List[RecallCandidate] = []
    candidate_count = 0
    exact_anchor_candidates = 0
    fallback_candidates = 0
    for row in rows:
        # REINFORCE is a durable feedback event for an existing canonical
        # Experience, not a second Capsule for Recall.
        if row.outcome == "REINFORCE":
            continue
        exact_hits = sum(value.casefold() in context_anchor_values for _, value in row.anchors)
        text_hit = bool(context.query.strip() and context.query.casefold() in row.searchable_text)
        if not exact_hits and row.experience_id not in search_scores.scores and not text_hit:
            continue
        candidate_count += 1
        exact_anchor_candidates += bool(exact_hits)
        fallback_candidates += bool(text_hit and row.experience_id not in search_scores.fts_ids)
        if effective[row.experience_id] != "verified" or not row.remote_verified:
            continue
        if not _applicable(row, context):
            continue
        feedback_score = row.success_count * 2 - row.failure_count
        confidence_score = {"high": 3, "medium": 2, "low": 1}.get(row.confidence, 0)
        score = (
            exact_hits * 100.0
            + search_scores.scores.get(row.experience_id, 0.0) * 10.0
            + (5.0 if text_hit else 0.0)
            + row.importance * 2.0
            + confidence_score
            + feedback_score
        )
        candidates.append(
            RecallCandidate(
                experience_id=row.experience_id,
                canonical_id=row.canonical_id,
                capsule=row.capsule,
                score=score,
                effective_experience_verification=effective[row.experience_id],
                remote_verified=row.remote_verified,
                exact_anchor_hits=exact_hits,
            )
        )
    candidates.sort(key=lambda item: (-item.score, item.canonical_id, item.experience_id))
    deduplicated: List[RecallCandidate] = []
    seen_canonical: Set[str] = set()
    for candidate in candidates:
        if candidate.canonical_id in seen_canonical:
            continue
        seen_canonical.add(candidate.canonical_id)
        deduplicated.append(candidate)
        if len(deduplicated) >= context.limit:
            break
    return RecallRun(
        candidates=tuple(deduplicated),
        stats=RecallStats(
            query_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
            candidate_count=candidate_count,
            capsule_count=len(deduplicated),
            approx_memory_tokens=sum(estimate_capsule_tokens(item.capsule) for item in deduplicated),
            full_markdown_expansions=0,
            exact_anchor_candidates=exact_anchor_candidates,
            fts_candidates=len(search_scores.fts_ids),
            trigram_candidates=len(search_scores.trigram_ids),
            fallback_candidates=fallback_candidates,
        ),
    )


def recall(connection: sqlite3.Connection, context: RecallContext) -> List[RecallCandidate]:
    """Backward-compatible Recall list API."""

    return list(recall_with_stats(connection, context).candidates)
