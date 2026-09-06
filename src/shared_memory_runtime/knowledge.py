"""Portable Skill Recall and Git-rebuildable Skill Evolution.

The portable knowledge layer is the authoritative source for this module. It
reads ``.knowledge/skills-index.jsonl`` and Markdown/YAML directly. SQLite,
local metrics, and Runtime caches are deliberately not consulted here.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - the optional Runtime declares PyYAML.
    yaml = None

from .recall import estimate_capsule_tokens


KNOWLEDGE_DIRNAME = ".knowledge"
SKILL_STATUSES = frozenset({"candidate", "verified"})
ALL_SKILL_STATUSES = frozenset({"candidate", "verified", "superseded", "invalid"})
EVOLUTION_DECISIONS = (
    "NO_CHANGE",
    "REINFORCE_SKILL",
    "CANDIDATE_SKILL",
    "UPDATE_SKILL",
    "CORRECT_SKILL",
)
REQUIRED_FIELDS = (
    "id",
    "kind",
    "path",
    "domain",
    "status",
    "version",
    "title",
    "summary",
    "capsule",
    "triggers",
    "applies_when",
    "does_not_apply_when",
    "search_anchors",
    "supporting_experience",
    "confidence",
    "created_at",
    "updated_at",
)
REQUIRED_HEADINGS = (
    "Goal",
    "Inputs",
    "Procedure",
    "Decision Points",
    "Search Strategy",
    "Validation",
    "Failure Patterns",
    "Exceptions",
    "Supporting Experience",
)
_RELATIVE_PATH_RE = re.compile(r"^(?!/)(?![A-Za-z]:[\\/])")


class KnowledgeContractError(ValueError):
    """A portable Skill, index, or evidence record violates the contract."""


def knowledge_root(codex_home: Optional[Path] = None) -> Path:
    """Resolve the Shared Knowledge repository from the current Agent home."""

    if codex_home is None:
        import os

        configured = os.environ.get("CODEX_HOME")
        codex_home = Path(configured).expanduser() if configured else Path.home() / ".codex"
    return codex_home / "skills"


def _index_path(shared_root: Path) -> Path:
    return shared_root / KNOWLEDGE_DIRNAME / "skills-index.jsonl"


def _is_safe_relative(path: object) -> bool:
    if not isinstance(path, str) or not path or not _RELATIVE_PATH_RE.match(path):
        return False
    parts = Path(path).parts
    return ".." not in parts


def _as_strings(value: object, field_name: str, *, required: bool = False) -> Tuple[str, ...]:
    if value is None and not required:
        return ()
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise KnowledgeContractError(f"{field_name} must be a collection of non-empty strings")
    result = tuple(str(item) for item in value)
    if len(set(result)) != len(result):
        raise KnowledgeContractError(f"{field_name} contains duplicates")
    return result


def _gate(value: object, field_name: str) -> Mapping[str, Tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise KnowledgeContractError(f"{field_name} must be a mapping")
    allowed = {"domains", "task_types", "required_anchors", "excluded_anchors"}
    unknown = set(value) - allowed
    if unknown:
        raise KnowledgeContractError(f"{field_name} has unknown keys: {sorted(unknown)}")
    return {
        key: _as_strings(value.get(key), f"{field_name}.{key}")
        for key in allowed
        if key in value
    }


def _evidence(value: object, field_name: str) -> Tuple[Mapping[str, object], ...]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise KnowledgeContractError(f"{field_name} must contain at least two evidence records")
    allowed = {
        "id",
        "path",
        "verification",
        "independence",
        "role",
        "result",
        "occurred_at",
    }
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            raise KnowledgeContractError(f"{field_name} entries must be mappings")
        missing = {"id", "path", "verification", "independence", "role", "result"} - set(item)
        if missing:
            raise KnowledgeContractError(f"{field_name} missing keys: {sorted(missing)}")
        if set(item) - allowed:
            raise KnowledgeContractError(f"{field_name} has unknown keys")
        if not isinstance(item["id"], str) or not item["id"].strip():
            raise KnowledgeContractError(f"{field_name}.id must be non-empty")
        if not _is_safe_relative(item["path"]):
            raise KnowledgeContractError(f"{field_name}.path must be repository-relative")
        if item["verification"] not in {"candidate", "verified", "unknown"}:
            raise KnowledgeContractError(f"{field_name}.verification is invalid")
        if item["independence"] not in {"independent", "dependent", "unknown"}:
            raise KnowledgeContractError(f"{field_name}.independence is invalid")
        if item["role"] not in {
            "candidate_support",
            "successful_reuse",
            "correction",
            "contradiction",
            "historical",
        }:
            raise KnowledgeContractError(f"{field_name}.role is invalid")
        if item["result"] not in {"verified", "success", "correction", "contradiction", "unknown"}:
            raise KnowledgeContractError(f"{field_name}.result is invalid")
        result.append(dict(item))
    return tuple(result)


def validate_skill_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    """Validate and normalize one index/front-matter Skill contract."""

    if not isinstance(metadata, Mapping):
        raise KnowledgeContractError("Skill metadata must be a mapping")
    missing = set(REQUIRED_FIELDS) - set(metadata)
    if missing:
        raise KnowledgeContractError(f"Skill metadata missing keys: {sorted(missing)}")
    if metadata.get("kind") != "skill":
        raise KnowledgeContractError("kind must be skill")
    if not _is_safe_relative(metadata.get("path")) or not str(metadata["path"]).startswith(
        ".knowledge/skills/"
    ) or not str(metadata["path"]).endswith(".md"):
        raise KnowledgeContractError("path must point to a portable .knowledge Skill Markdown file")
    if metadata.get("status") not in ALL_SKILL_STATUSES:
        raise KnowledgeContractError("status is invalid")
    if not isinstance(metadata.get("id"), str) or not re.fullmatch(
        r"[a-z0-9]+(?:\.[a-z0-9-]+)+", str(metadata["id"])
    ):
        raise KnowledgeContractError("id is invalid")
    for field_name in ("domain", "version", "title", "summary", "capsule", "confidence", "created_at", "updated_at"):
        if not isinstance(metadata.get(field_name), str) or not metadata[field_name].strip():
            raise KnowledgeContractError(f"{field_name} must be a non-empty string")
    if metadata["confidence"] not in {"low", "medium", "high"}:
        raise KnowledgeContractError("confidence is invalid")
    _as_strings(metadata["triggers"], "triggers", required=True)
    _as_strings(metadata["search_anchors"], "search_anchors")
    _gate(metadata["applies_when"], "applies_when")
    _gate(metadata["does_not_apply_when"], "does_not_apply_when")
    _evidence(metadata["supporting_experience"], "supporting_experience")
    return dict(metadata)


def read_skill_index(shared_root: Path) -> Tuple[Mapping[str, object], ...]:
    """Read the hand-maintained JSONL index without a generated intermediary."""

    path = _index_path(shared_root)
    if not path.is_file():
        raise KnowledgeContractError(f"portable index missing: {path}")
    records: List[Mapping[str, object]] = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise KnowledgeContractError(f"invalid JSONL at line {line_number}") from exc
        metadata = validate_skill_metadata(record)
        if metadata["id"] in seen:
            raise KnowledgeContractError(f"duplicate Skill id: {metadata['id']}")
        seen.add(metadata["id"])
        records.append(metadata)
    return tuple(records)


def _parse_markdown(path: Path) -> Tuple[Mapping[str, object], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise KnowledgeContractError(f"missing YAML front matter: {path}")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise KnowledgeContractError(f"unterminated YAML front matter: {path}")
    raw = text[4:end]
    if yaml is None:
        try:
            metadata = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise KnowledgeContractError("PyYAML is required for non-JSON front matter") from exc
    else:
        # BaseLoader preserves ISO timestamps as strings, matching JSONL and
        # keeping the portable contract free from Python-specific date types.
        metadata = yaml.load(raw, Loader=yaml.BaseLoader)
    if not isinstance(metadata, Mapping):
        raise KnowledgeContractError(f"front matter is not a mapping: {path}")
    body = text[end + len("\n---\n") :]
    headings = tuple(
        line[2:].strip()
        for line in body.splitlines()
        if line.startswith("# ")
    )
    if headings[: len(REQUIRED_HEADINGS)] != REQUIRED_HEADINGS:
        raise KnowledgeContractError(f"body headings do not match contract: {path}")
    return validate_skill_metadata(metadata), body


def load_skill(shared_root: Path, record: Mapping[str, object]) -> Tuple[Mapping[str, object], str, Path]:
    """Load and validate the Markdown Skill named by one index record."""

    path_value = record.get("path")
    if not _is_safe_relative(path_value):
        raise KnowledgeContractError(f"index path is not repository-relative: {path_value}")
    path = shared_root / str(path_value)
    metadata, body = _parse_markdown(path)
    if metadata != dict(record):
        differing = sorted(key for key in set(metadata) | set(record) if metadata.get(key) != record.get(key))
        raise KnowledgeContractError(f"index and Skill disagree for {record.get('id')}: {differing}")
    return metadata, body, path


def _lower_strings(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(value.casefold() for value in values)


def _context_text(context: "SkillRecallContext") -> str:
    return " ".join((context.task, *context.anchors)).casefold()


def _gate_applies(record: Mapping[str, object], context: "SkillRecallContext") -> bool:
    text = _context_text(context)
    anchors = _lower_strings(context.anchors)
    applies = record["applies_when"]
    excludes = record["does_not_apply_when"]
    for key, actual in (("domains", context.domain), ("task_types", context.task_type)):
        allowed = _lower_strings(applies.get(key, ()))
        if allowed and actual is not None and actual.casefold() not in allowed:
            return False
        if allowed and actual is None:
            continue
        denied = _lower_strings(excludes.get(key, ()))
        if actual is not None and actual.casefold() in denied:
            return False
    required = _lower_strings(applies.get("required_anchors", ()))
    if any(anchor not in text and anchor not in anchors for anchor in required):
        return False
    excluded = _lower_strings(
        (*excludes.get("excluded_anchors", ()), *excludes.get("required_anchors", ()))
    )
    if any(anchor in text or anchor in anchors for anchor in excluded):
        return False
    return True


def _recency(value: object) -> float:
    if not isinstance(value, str):
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _successful_reuse_count(record: Mapping[str, object]) -> int:
    return sum(
        1
        for evidence in record["supporting_experience"]
        if evidence.get("role") == "successful_reuse"
        and evidence.get("result") == "success"
        and evidence.get("verification") == "verified"
        and evidence.get("independence") == "independent"
    )


@dataclass(frozen=True)
class SkillRecallContext:
    task: str = ""
    domain: Optional[str] = None
    task_type: Optional[str] = None
    anchors: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillRecallCandidate:
    skill_id: str
    path: str
    status: str
    capsule: str
    score: Tuple[int, ...]
    exact_anchor_hits: int
    exact_trigger_hits: int
    domain_match: bool
    task_type_match: bool
    applicability_passed: bool
    historical_reuse_successes: int


@dataclass(frozen=True)
class ExperienceExpansion:
    experience_id: str
    path: str
    verification: str
    result: str
    capsule: str


@dataclass(frozen=True)
class SkillRecallStats:
    query_ms: float
    candidate_count: int
    capsule_count: int
    full_skill_count: int
    experience_expansion_count: int
    approx_injected_knowledge_tokens: int
    expansion_reason: Optional[str] = None


@dataclass(frozen=True)
class SkillRecallRun:
    candidates: Tuple[SkillRecallCandidate, ...]
    full_skills: Tuple[str, ...] = ()
    experiences: Tuple[ExperienceExpansion, ...] = ()
    stats: SkillRecallStats = field(
        default_factory=lambda: SkillRecallStats(0.0, 0, 0, 0, 0, 0)
    )

    def as_mapping(self) -> Mapping[str, object]:
        return {
            "skills": [
                {
                    "id": item.skill_id,
                    "path": item.path,
                    "status": item.status,
                    "capsule": item.capsule,
                    "score": item.score,
                    "applicability_passed": item.applicability_passed,
                }
                for item in self.candidates
            ],
            "full_skills": list(self.full_skills),
            "experiences": [item.__dict__ for item in self.experiences],
            "stats": self.stats.__dict__,
        }


def _experience_capsule(path: Path) -> Tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return "", "unknown"
    end = text.find("\n---\n", 4)
    if end < 0 or yaml is None:
        return "", "unknown"
    metadata = yaml.load(text[4:end], Loader=yaml.BaseLoader)
    if not isinstance(metadata, Mapping):
        return "", "unknown"
    experience = metadata.get("experience", {})
    if not isinstance(experience, Mapping):
        return "", "unknown"
    return str(experience.get("capsule", "")), str(experience.get("verification", "unknown"))


def _load_experience_expansions(
    shared_root: Path,
    records: Sequence[Mapping[str, object]],
    *,
    limit: int = 2,
) -> Tuple[ExperienceExpansion, ...]:
    result = []
    seen = set()
    for record in records:
        for evidence in record["supporting_experience"]:
            if evidence["id"] in seen or evidence["verification"] != "verified":
                continue
            path = shared_root / evidence["path"]
            if not path.is_file():
                continue
            capsule, verification = _experience_capsule(path)
            result.append(
                ExperienceExpansion(
                    experience_id=str(evidence["id"]),
                    path=str(evidence["path"]),
                    verification=verification,
                    result=str(evidence["result"]),
                    capsule=capsule,
                )
            )
            seen.add(evidence["id"])
            if len(result) >= limit:
                return tuple(result)
    return tuple(result)


def recall_skills(
    shared_root: Path,
    context: SkillRecallContext,
    *,
    full_skill_ids: Sequence[str] = (),
    expand_experiences: bool = False,
) -> SkillRecallRun:
    """Execute authoritative portable Recall with bounded expansion."""

    started = time.perf_counter_ns()
    records = read_skill_index(shared_root)
    valid_records = [record for record in records if record["status"] in SKILL_STATUSES]
    candidates = []
    context_anchors = {value.casefold() for value in context.anchors}
    task_text = context.task.casefold()
    for record in valid_records:
        if not _gate_applies(record, context):
            continue
        anchors = tuple(str(value) for value in record["search_anchors"])
        exact_anchor_hits = sum(value.casefold() in context_anchors for value in anchors)
        exact_trigger_hits = sum(value.casefold() in task_text for value in record["triggers"])
        domain_match = context.domain is not None and context.domain.casefold() == str(record["domain"]).casefold()
        applies_types = {value.casefold() for value in record["applies_when"].get("task_types", ())}
        task_type_match = context.task_type is not None and context.task_type.casefold() in applies_types
        reuse_count = _successful_reuse_count(record)
        confidence = {"high": 3, "medium": 2, "low": 1}[str(record["confidence"])]
        score = (
            exact_anchor_hits,
            exact_trigger_hits,
            int(domain_match),
            int(task_type_match),
            1,
            int(record["status"] == "verified"),
            reuse_count,
            confidence,
            int(_recency(record["updated_at"])),
        )
        candidates.append(
            SkillRecallCandidate(
                skill_id=str(record["id"]),
                path=str(record.get("path", "")),
                status=str(record["status"]),
                capsule=str(record["capsule"]),
                score=score,
                exact_anchor_hits=exact_anchor_hits,
                exact_trigger_hits=exact_trigger_hits,
                domain_match=bool(domain_match),
                task_type_match=bool(task_type_match),
                applicability_passed=True,
                historical_reuse_successes=reuse_count,
            )
        )
    candidates.sort(key=lambda item: (tuple(-value for value in item.score), item.skill_id))
    selected = tuple(candidates[:4])
    by_id = {str(record["id"]): record for record in valid_records}
    requested = tuple(dict.fromkeys(str(item) for item in full_skill_ids))
    applicable_by_id = {
        str(record["id"]): record
        for record in valid_records
        if _gate_applies(record, context)
    }
    full_ids = tuple(item for item in requested if item in applicable_by_id)[:2]
    full_paths = []
    for skill_id in full_ids:
        record = applicable_by_id[skill_id]
        _, body, _ = load_skill(shared_root, record)
        full_paths.append(body)
    experiences: Tuple[ExperienceExpansion, ...] = ()
    if expand_experiences or not selected:
        evidence_records = [by_id[item.skill_id] for item in selected] if selected else valid_records
        experiences = _load_experience_expansions(shared_root, evidence_records, limit=2)
    token_count = sum(estimate_capsule_tokens(item.capsule) for item in selected)
    token_count += sum(estimate_capsule_tokens(item.capsule) for item in experiences)
    token_count += sum(max(1, len(body) // 4) for body in full_paths)
    reason = None
    if token_count > 600:
        reason = "explicit expansion exceeded the 200-600 token guidance target"
    return SkillRecallRun(
        candidates=selected,
        full_skills=tuple(full_paths),
        experiences=experiences,
        stats=SkillRecallStats(
            query_ms=(time.perf_counter_ns() - started) / 1_000_000,
            candidate_count=len(candidates),
            capsule_count=len(selected),
            full_skill_count=len(full_paths),
            experience_expansion_count=len(experiences),
            approx_injected_knowledge_tokens=token_count,
            expansion_reason=reason,
        ),
    )


def validate_knowledge_tree(shared_root: Path, *, require_git_tracked: bool = False) -> Tuple[str, ...]:
    """Validate index/Skill/evidence parity and return deterministic findings."""

    findings: List[str] = []
    records = read_skill_index(shared_root)
    for record in records:
        metadata, _, path = load_skill(shared_root, record)
        if metadata["id"] != record["id"]:
            raise KnowledgeContractError(f"metadata/index id mismatch: {record['id']}")
        for evidence in record["supporting_experience"]:
            evidence_path = shared_root / str(evidence["path"])
            if not evidence_path.is_file():
                raise KnowledgeContractError(f"missing supporting Experience: {evidence['path']}")
            if require_git_tracked:
                check = subprocess.run(
                    ["git", "ls-files", "--error-unmatch", str(evidence["path"])],
                    cwd=shared_root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if check.returncode != 0:
                    raise KnowledgeContractError(f"supporting Experience is not Git-tracked: {evidence['path']}")
        if not _is_safe_relative(str(path.relative_to(shared_root))):
            raise KnowledgeContractError(f"Skill path is not portable: {path}")
    forbidden = list((shared_root / KNOWLEDGE_DIRNAME).rglob("SKILL.md"))
    if forbidden:
        raise KnowledgeContractError(".knowledge must not contain SKILL.md")
    findings.append(f"skills={len(records)}")
    findings.append("index=direct-readable")
    findings.append("evidence=git-rebuildable")
    return tuple(findings)


@dataclass(frozen=True)
class SkillEvolutionResult:
    skill_id: str
    current_status: str
    decision: str
    candidate_support_count: int
    successful_reuse_count: int
    contradiction_count: int
    evidence_source: str = "git-tracked-shared-knowledge"

    def as_mapping(self) -> Mapping[str, object]:
        return self.__dict__


def skill_evolution_check(shared_root: Path, skill_id: str) -> SkillEvolutionResult:
    """Compute only the frozen evolution enum from portable evidence."""

    record = next((item for item in read_skill_index(shared_root) if item["id"] == skill_id), None)
    if record is None:
        raise KnowledgeContractError(f"unknown Skill: {skill_id}")
    support = record["supporting_experience"]
    candidate_support = {
        item["id"]
        for item in support
        if item["role"] == "candidate_support"
        and item["verification"] == "verified"
        and item["independence"] == "independent"
        and item["result"] == "verified"
    }
    successful_reuse = {
        item["id"]
        for item in support
        if item["role"] == "successful_reuse"
        and item["verification"] == "verified"
        and item["independence"] == "independent"
        and item["result"] == "success"
    }
    contradictions = {
        item["id"]
        for item in support
        if item["role"] == "contradiction" or item["result"] == "contradiction"
    }
    status = str(record["status"])
    if contradictions and status == "verified":
        decision = "CORRECT_SKILL"
    elif status == "candidate" and len(candidate_support) >= 2 and len(successful_reuse) >= 2:
        decision = "UPDATE_SKILL"
    elif status == "candidate" and len(candidate_support) >= 2:
        decision = "CANDIDATE_SKILL"
    elif status == "verified" and len(successful_reuse) >= 2 and not contradictions:
        decision = "REINFORCE_SKILL"
    else:
        decision = "NO_CHANGE"
    return SkillEvolutionResult(
        skill_id=str(record["id"]),
        current_status=status,
        decision=decision,
        candidate_support_count=len(candidate_support),
        successful_reuse_count=len(successful_reuse),
        contradiction_count=len(contradictions),
    )


__all__ = [
    "ALL_SKILL_STATUSES",
    "EVOLUTION_DECISIONS",
    "ExperienceExpansion",
    "KnowledgeContractError",
    "REQUIRED_HEADINGS",
    "SkillEvolutionResult",
    "SkillRecallCandidate",
    "SkillRecallContext",
    "SkillRecallRun",
    "SkillRecallStats",
    "knowledge_root",
    "load_skill",
    "read_skill_index",
    "recall_skills",
    "skill_evolution_check",
    "validate_knowledge_tree",
    "validate_skill_metadata",
]
