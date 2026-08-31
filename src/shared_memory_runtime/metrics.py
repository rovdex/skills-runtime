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

    def _validate(self) -> None:
        if not self.task_id.strip() or not self.task_result.strip():
            raise ValueError("task_id and task_result are required")
        if not self.started_at.strip() or not self.finished_at.strip():
            raise ValueError("started_at and finished_at are required")
        for name in ("recall_ms", "compiler_ms", "finalization_ms"):
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
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.extra_learning_calls == 0 and self.extra_learning_reason_codes:
            raise ValueError("extra learning reasons require an extra learning call")
        unknown = set(self.extra_learning_reason_codes) - EXTRA_LEARNING_REASON_CODES
        if unknown:
            raise ValueError(f"unknown extra learning reason code: {sorted(unknown)}")

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
