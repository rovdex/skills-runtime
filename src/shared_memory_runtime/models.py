"""Domain values shared by Markdown parsing, projection, Recall, and Compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Tuple


def _tuple_values(value: object) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple, set)):
        raise ValueError("applicability values must be lists or strings")
    return tuple(str(item).strip() for item in value if str(item).strip())


@dataclass(frozen=True)
class Applicability:
    project_keys: Tuple[str, ...] = ()
    task_kinds: Tuple[str, ...] = ()
    required_anchors: Tuple[str, ...] = ()
    excluded_project_keys: Tuple[str, ...] = ()
    excluded_task_kinds: Tuple[str, ...] = ()
    excluded_anchors: Tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, object]]) -> "Applicability":
        mapping = value or {}
        applies = mapping.get("applies_when") or {}
        does_not_apply = mapping.get("does_not_apply_when") or {}
        if not isinstance(applies, Mapping) or not isinstance(does_not_apply, Mapping):
            raise ValueError("applicability sections must be mappings")
        return cls(
            project_keys=_tuple_values(applies.get("project_keys")),
            task_kinds=_tuple_values(applies.get("task_kinds")),
            required_anchors=_tuple_values(applies.get("required_anchors")),
            excluded_project_keys=_tuple_values(does_not_apply.get("project_keys")),
            excluded_task_kinds=_tuple_values(does_not_apply.get("task_kinds")),
            excluded_anchors=_tuple_values(does_not_apply.get("excluded_anchors")),
        )

    def as_mapping(self) -> Mapping[str, Mapping[str, Tuple[str, ...]]]:
        return {
            "applies_when": {
                "project_keys": self.project_keys,
                "task_kinds": self.task_kinds,
                "required_anchors": self.required_anchors,
            },
            "does_not_apply_when": {
                "project_keys": self.excluded_project_keys,
                "task_kinds": self.excluded_task_kinds,
                "excluded_anchors": self.excluded_anchors,
            },
        }

    def semantic_key(self) -> Tuple[Tuple[str, ...], ...]:
        return (
            tuple(sorted(self.project_keys)),
            tuple(sorted(self.task_kinds)),
            tuple(sorted(self.required_anchors)),
            tuple(sorted(self.excluded_project_keys)),
            tuple(sorted(self.excluded_task_kinds)),
            tuple(sorted(self.excluded_anchors)),
        )


@dataclass(frozen=True)
class FeedbackEvent:
    experience_id: str
    canonical_id: str
    reused: bool
    result: str
    used_at: str


@dataclass(frozen=True)
class ExperienceRecord:
    id: str
    task_id: str
    scope: str
    project_key: Optional[str]
    kind: str
    fragment_status: str
    experience_verification: str
    title: str
    summary: str
    capsule: str
    confidence: str
    importance: int
    source_path: str
    source_hash: str
    created_at: str
    updated_at: Optional[str]
    outcome: str
    canonical_id: str
    applicability: Applicability = field(default_factory=Applicability)
    trigger: str = ""
    core_action: str = ""
    does_not_apply: str = ""
    exceptions: str = ""
    anchors: Tuple[Tuple[str, str], ...] = ()
    supersedes: Tuple[str, ...] = ()
    feedback: Optional[FeedbackEvent] = None

    @property
    def effective_text(self) -> str:
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
        )

    def anchor_values(self) -> Tuple[str, ...]:
        return tuple(value for _, value in self.anchors)


def normalized_tokens(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted({str(value).strip().casefold() for value in values if str(value).strip()}))
