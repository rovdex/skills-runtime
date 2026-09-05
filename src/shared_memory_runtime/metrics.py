"""Best-effort, local-only terminal performance metrics."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


EXTRA_LEARNING_REASON_CODES = {
    "semantic_conflict",
    "new_correct_ambiguity",
    "complex_applicability_ambiguity",
}


def metrics_path(codex_home: Optional[Path] = None) -> Path:
    configured_home = codex_home or os.environ.get("CODEX_HOME")
    root = Path(configured_home).expanduser() if configured_home else Path.home() / ".codex"
    return root / ".state" / "experience-runtime" / "metrics.jsonl"


@dataclass(frozen=True)
class TaskMetrics:
    task_id: str
    project_key: Optional[str]
    task_result: str
    started_at: str
    finished_at: str
    recall_ms: float = 0.0
    candidate_count: int = 0
    capsule_count: int = 0
    approx_memory_tokens: int = 0
    full_markdown_expansions: int = 0
    terminal_model_calls: int = 0
    extra_learning_calls: int = 0
    extra_learning_reason_codes: Tuple[str, ...] = field(default_factory=tuple)
    compiler_ms: float = 0.0
    markdown_reads: int = 0
    markdown_writes: int = 0
    state_writes: int = 0
    sqlite_transactions: int = 0
    shared_knowledge_commits: int = 0
    pushes: int = 0
    remote_verifies: int = 0
    finalization_ms: float = 0.0
    policy_compile_ms: float = 0.0
    policy_task_class: Optional[str] = None
    policy_valid: Optional[bool] = None
    policy_fallback: Optional[bool] = None
    policy_skill_count: int = 0
    policy_max_capsules: Optional[int] = None
    policy_max_read_lines: Optional[int] = None
    policy_max_output_kb: Optional[int] = None
    policy_shadow_differences: int = 0
    policy_mode: Optional[str] = None
    policy_applied_fields: Tuple[str, ...] = field(default_factory=tuple)
    policy_source: Optional[str] = None
    policy_observed_actual_behavior: Dict[str, int] = field(default_factory=dict)
    policy_protocol_consumed_fields: Tuple[str, ...] = field(default_factory=tuple)
    policy_not_instrumented_fields: Tuple[str, ...] = field(default_factory=tuple)
    policy_runtime_enforced_fields: Tuple[str, ...] = field(default_factory=tuple)
    experience_advice_count: int = 0
    adapted_field_count: int = 0
    experience_conflict_count: int = 0
    experience_policy_sources: Tuple[str, ...] = field(default_factory=tuple)
    jit_policy_evidence_version: Optional[str] = None

    def _validate(self) -> None:
        if not self.task_id.strip() or not self.task_result.strip():
            raise ValueError("task_id and task_result are required")
        if not self.started_at.strip() or not self.finished_at.strip():
            raise ValueError("started_at and finished_at are required")
        for name in ("recall_ms", "compiler_ms", "finalization_ms", "policy_compile_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        for name in (
            "candidate_count",
            "capsule_count",
            "approx_memory_tokens",
            "full_markdown_expansions",
            "terminal_model_calls",
            "extra_learning_calls",
            "markdown_reads",
            "markdown_writes",
            "state_writes",
            "sqlite_transactions",
            "shared_knowledge_commits",
            "pushes",
            "remote_verifies",
            "policy_skill_count",
            "policy_shadow_differences",
            "experience_advice_count",
            "adapted_field_count",
            "experience_conflict_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.extra_learning_calls == 0 and self.extra_learning_reason_codes:
            raise ValueError("extra learning reasons require an extra learning call")
        unknown = set(self.extra_learning_reason_codes) - EXTRA_LEARNING_REASON_CODES
        if unknown:
            raise ValueError(f"unknown extra learning reason code: {sorted(unknown)}")
        for name in ("policy_max_capsules", "policy_max_read_lines", "policy_max_output_kb"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or null")
        if self.policy_mode is not None and self.policy_mode not in {"shadow", "adapted", "active"}:
            raise ValueError("policy_mode must be shadow, adapted, active, or null")
        if self.policy_source is not None and self.policy_source not in {
            "existing_default",
            "validated_jit_task_policy",
            "skill_default",
            "project_requirement",
            "user_requirement",
        }:
            raise ValueError("unknown policy_source")
        if self.jit_policy_evidence_version is not None and not self.jit_policy_evidence_version.strip():
            raise ValueError("jit_policy_evidence_version must be non-empty or null")
        for value in self.experience_policy_sources:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("experience_policy_sources must contain non-empty strings")
        for key, value in self.policy_observed_actual_behavior.items():
            if key not in {"shell_calls", "wsl_process_count", "file_reads", "tool_output_bytes", "capsule_count"}:
                raise ValueError("unknown observed actual behavior field")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("observed actual behavior values must be non-negative integers")

    def as_mapping(self) -> Dict[str, object]:
        self._validate()
        return {
            "task_id": self.task_id,
            "project_key": self.project_key,
            "task_result": self.task_result,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "recall_ms": float(self.recall_ms),
            "candidate_count": self.candidate_count,
            "capsule_count": self.capsule_count,
            "approx_memory_tokens": self.approx_memory_tokens,
            "full_markdown_expansions": self.full_markdown_expansions,
            "terminal_model_calls": self.terminal_model_calls,
            "extra_learning_calls": self.extra_learning_calls,
            "extra_learning_reason_codes": sorted(set(self.extra_learning_reason_codes)),
            "compiler_ms": float(self.compiler_ms),
            "markdown_reads": self.markdown_reads,
            "markdown_writes": self.markdown_writes,
            "state_writes": self.state_writes,
            "sqlite_transactions": self.sqlite_transactions,
            "shared_knowledge_commits": self.shared_knowledge_commits,
            "pushes": self.pushes,
            "remote_verifies": self.remote_verifies,
            "finalization_ms": float(self.finalization_ms),
            "policy_compile_ms": float(self.policy_compile_ms),
            "policy_task_class": self.policy_task_class,
            "policy_valid": self.policy_valid,
            "policy_fallback": self.policy_fallback,
            "policy_skill_count": self.policy_skill_count,
            "policy_max_capsules": self.policy_max_capsules,
            "policy_max_read_lines": self.policy_max_read_lines,
            "policy_max_output_kb": self.policy_max_output_kb,
            "policy_shadow_differences": self.policy_shadow_differences,
            "policy_mode": self.policy_mode,
            "policy_applied_fields": sorted(set(self.policy_applied_fields)),
            "policy_source": self.policy_source,
            "policy_observed_actual_behavior": dict(sorted(self.policy_observed_actual_behavior.items())),
            "policy_protocol_consumed_fields": sorted(set(self.policy_protocol_consumed_fields)),
            "policy_not_instrumented_fields": sorted(set(self.policy_not_instrumented_fields)),
            "policy_runtime_enforced_fields": sorted(set(self.policy_runtime_enforced_fields)),
            "experience_advice_count": self.experience_advice_count,
            "adapted_field_count": self.adapted_field_count,
            "experience_conflict_count": self.experience_conflict_count,
            "experience_policy_sources": sorted(set(self.experience_policy_sources)),
            "jit_policy_evidence_version": self.jit_policy_evidence_version,
        }


def record_task_metrics(metrics: TaskMetrics, codex_home: Optional[Path] = None) -> bool:
    """Append one whitelisted JSON record; all failures are non-blocking."""

    try:
        output_path = metrics_path(codex_home)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(metrics.as_mapping(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        descriptor = os.open(str(output_path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
        return True
    except (OSError, TypeError, UnicodeError, ValueError):
        return False


def record_final_jit_policy_metrics(
    metrics: TaskMetrics, codex_home: Optional[Path] = None
) -> bool:
    """Record one final JIT evidence row for a task, idempotently.

    This reuses the existing local JSONL store.  The explicit version marker
    distinguishes the final JIT row from ordinary task metrics; a second
    identical write is accepted while a conflicting duplicate is rejected.
    Formal Task Single Writer ownership remains the concurrency boundary.
    """

    if metrics.jit_policy_evidence_version is None:
        return False
    try:
        output_path = metrics_path(codex_home)
        expected = metrics.as_mapping()
        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if (
                        isinstance(row, dict)
                        and row.get("task_id") == metrics.task_id
                        and row.get("jit_policy_evidence_version") is not None
                    ):
                        return row == expected
        output_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        descriptor = os.open(str(output_path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
        return True
    except (OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError):
        return False
