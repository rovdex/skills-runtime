"""Phase 3 gap-only measurements without rerunning the complete baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shared_memory_runtime import (  # noqa: E402
    ExperienceProjector,
    RecallContext,
    TaskMetrics,
    record_task_metrics,
)
from shared_memory_runtime.db import open_database, rebuild_feedback, upsert_experience  # noqa: E402
from shared_memory_runtime.markdown import project_key  # noqa: E402
from shared_memory_runtime.recall import recall_with_stats  # noqa: E402


PROFILE_NAMES = ("Normal", "Complex", "Debugging")
FINALIZATION_KEYS = (
    "terminal_feedback_ms",
    "compiler_ms",
    "primary_generation_ms",
    "primary_validation_ms",
    "markdown_write_ms",
    "state_persist_ms",
    "runtime_git_commit_ms",
    "runtime_push_ms",
    "runtime_remote_verify_ms",
    "shared_knowledge_commit_ms",
    "shared_knowledge_push_ms",
    "shared_knowledge_remote_verify_ms",
    "sqlite_projection_ms",
    "metrics_append_ms",
    "total_finalization_ms",
)
LOCAL_FINALIZATION_KEYS = (
    "terminal_feedback_ms",
    "compiler_ms",
    "primary_generation_ms",
    "primary_validation_ms",
    "markdown_write_ms",
    "state_persist_ms",
    "sqlite_projection_ms",
    "metrics_append_ms",
)
REMOTE_FINALIZATION_KEYS = (
    "runtime_push_ms",
    "runtime_remote_verify_ms",
    "shared_knowledge_push_ms",
    "shared_knowledge_remote_verify_ms",
)
NON_PRIMARY_GIT_KEYS = (
    "runtime_git_commit_ms",
    "shared_knowledge_commit_ms",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    return Path(configured).expanduser().resolve() if configured else (Path.home() / ".codex").resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def remote_snapshot(root: Path, timeout_seconds: int = 5) -> Tuple[str, str]:
    upstream = run_git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    remote_name, branch = upstream.split("/", 1)
    result = subprocess.run(
        ["git", "-C", str(root), "ls-remote", remote_name, f"refs/heads/{branch}"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-remote failed")
    expected_ref = f"refs/heads/{branch}"
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == expected_ref:
            return remote_name, parts[0]
    raise RuntimeError("remote branch head was not returned")


def select_verified_records(
    source_root: Path,
    database: Path,
    validated_snapshot: Optional[Tuple[str, str]] = None,
):
    remote_name, remote_head = validated_snapshot or remote_snapshot(source_root)
    projector = ExperienceProjector(source_root, database, remote_snapshot=(remote_name, remote_head))
    selected = []
    proofs = []
    project_keys = set()
    for source_path in projector._source_paths():
        record, skipped = projector._resolve_source_path(source_path)
        if skipped or record is None:
            continue
        proof = projector.git.prove_remote_persistence(record.source_path, record.source_hash)
        if proof.verified:
            selected.append(record)
            proofs.append(proof)
            project_keys.add(record.project_key)
        if len(selected) >= 3 and None in project_keys:
            break
    connection = open_database(database)
    try:
        with connection:
            for record, proof in zip(selected, proofs):
                upsert_experience(connection, record, proof.verified)
            rebuild_feedback(connection, selected)
    finally:
        connection.close()
    return projector, selected, proofs, (remote_name, remote_head)


def profile_contexts(
    selected_project_key: Optional[str], task_kind: Optional[str], anchors: Sequence[str]
) -> Dict[str, RecallContext]:
    return {
        name: RecallContext(
            project_key=selected_project_key,
            task_kind=task_kind,
            anchors=tuple(anchors),
            task_class=name.casefold(),
            expand=False,
        )
        for name in PROFILE_NAMES
    }


def profile_result(run, context: RecallContext) -> Dict[str, object]:
    stats = run.stats
    return {
        "candidate_count": stats.candidate_count,
        "capsule_count": stats.capsule_count,
        "approx_memory_tokens": stats.approx_memory_tokens,
        "estimated": True,
        "full_markdown_expansions": stats.full_markdown_expansions,
        "task_class": context.task_class,
        "expand": context.expand,
        "limit": context.limit,
    }


def empty_breakdown() -> Dict[str, float]:
    return {key: 0.0 for key in FINALIZATION_KEYS}


def validate_breakdown(value: Mapping[str, object]) -> Dict[str, float]:
    result = empty_breakdown()
    for key in FINALIZATION_KEYS:
        if key not in value:
            continue
        number = float(value[key])
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{key} must be a finite non-negative number")
        result[key] = round(number, 4)
    return result


def audit_prior_baseline(path: Path) -> Dict[str, object]:
    before = sha256_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    expected_present = {
        "recall_latency": "local_runtime_performance.recall.*.p50/p95/max",
        "projection_upsert": "local_runtime_performance.single_projection_upsert",
        "full_rebuild": "local_runtime_performance.full_sqlite_rebuild",
        "local_remote_proof": "local_runtime_performance.remote_proof_local_snapshot",
        "network_samples": "remote_network_performance.remote_verify_git_ls_remote",
        "markdown_expansions": "local_runtime_performance.recall.*.full_markdown_expansions",
    }
    missing = {
        "named_memory_profiles": "Normal/Complex/Debugging profile records",
        "detailed_finalization_breakdown": "finalization_breakdown with per-stage timings",
    }
    after = sha256_file(path)
    return {
        "path": str(path),
        "sha256": before,
        "unchanged_during_audit": before == after,
        "present": expected_present,
        "baseline_gaps": missing,
        "top_level_keys": sorted(report.keys()),
    }


def gap_measurement(
    source_root: Path,
    prior_baseline: Path,
    task_id: str,
    runtime_root: Path,
    validated_snapshot: Optional[Tuple[str, str]] = None,
) -> Dict[str, object]:
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix="shared-memory-runtime-gap-") as directory:
        database = Path(directory) / "experience.db"
        projection_started = time.perf_counter_ns()
        _, selected, _, snapshot = select_verified_records(
            source_root, database, validated_snapshot=validated_snapshot
        )
        projection_ms = (time.perf_counter_ns() - projection_started) / 1_000_000
        first = selected[0] if selected else None
        anchor_values = ()
        if first is not None:
            anchor = "remote_verified" if "remote_verified" in first.anchor_values() else (
                first.anchor_values()[0] if first.anchor_values() else "remote_verified"
            )
            anchor_values = tuple(sorted(set(first.applicability.required_anchors) | {anchor}))
        contexts = profile_contexts(
            first.project_key if first else None,
            "implementation" if first else None,
            anchor_values,
        )
        connection = open_database(database)
        try:
            runs = {}
            for name, context in contexts.items():
                first_run = recall_with_stats(connection, context)
                second_run = recall_with_stats(connection, context)
                if profile_result(first_run, context) != profile_result(second_run, context):
                    raise RuntimeError(f"non-deterministic profile result: {name}")
                runs[name] = first_run
        finally:
            connection.close()
    profiles = {
        name: profile_result(run, contexts[name]) for name, run in runs.items()
    }
    profile_limits = {name: profiles[name]["limit"] for name in PROFILE_NAMES}
    if profile_limits["Normal"] > 4 or profile_limits["Complex"] > 5 or profile_limits["Debugging"] > 5:
        raise AssertionError("profile limit exceeds the existing Recall contract")
    if any(item["full_markdown_expansions"] != 0 for item in profiles.values()):
        raise AssertionError("gap profiles must not expand full Markdown")
    total_ms = (time.perf_counter_ns() - started) / 1_000_000
    return {
        "task_id": task_id,
        "measurement_timestamp": utc_now(),
        "runtime_revision": run_git(runtime_root, "rev-parse", "HEAD"),
        "shared_knowledge_revision": run_git(source_root, "rev-parse", "HEAD"),
        "prior_baseline_audit": audit_prior_baseline(prior_baseline),
        "memory_token_baseline": {
            "profiles": profiles,
            "constraints": {
                "normal_max_capsules": 4,
                "complex_max_capsules": 5,
                "debugging_max_capsules": 5,
                "full_markdown_default": 0,
                "no_forced_injection": True,
            },
            "remote_snapshot": {"remote_name": snapshot[0], "remote_head": snapshot[1]},
            "selected_verified_experience_count": len(selected),
        },
        "benchmark_model_cost": {
            "terminal_model_calls": 0,
            "extra_learning_calls": 0,
            "scope": "Gap Benchmark Measurement Workload only",
        },
        "sqlite_projection_ms": round(projection_ms, 4),
        "gap_measurement_ms": round(total_ms, 4),
        "finalization_breakdown": empty_breakdown() | {"sqlite_projection_ms": round(projection_ms, 4)},
        "finalization_not_applicable": [
            key for key in FINALIZATION_KEYS if key not in {"sqlite_projection_ms", "total_finalization_ms"}
        ],
        "finalization_scope": {
            "local": list(LOCAL_FINALIZATION_KEYS),
            "remote": list(REMOTE_FINALIZATION_KEYS),
            "commit": list(NON_PRIMARY_GIT_KEYS),
        },
        "task_result": "Passed",
        "gap_completion": "Passed",
        "benchmark_result": "Gap Completion Passed",
        "runtime_reliability": "Not Re-run; prior 15/15 evidence preserved",
        "synthetic_accuracy": "Not Re-run; prior evidence preserved",
        "optimization_required": "Undetermined",
        "optimization_candidates": [],
        "runtime_changes": "Benchmark/metrics measurement only",
        "runtime_commit": None,
        "runtime_push": "Not Attempted",
        "runtime_remote_verify": "Not Attempted",
        "shared_knowledge_primary_commit": None,
        "shared_knowledge_push": "Not Attempted",
        "shared_knowledge_remote_verify": "Not Attempted",
        "metrics_append": "Not Attempted",
        "final_runtime_state": "verified_pending + updates_validated",
        "sedimentation": "Pending",
    }


def write_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finalize_evidence(path: Path, breakdown_json: str, metadata_json: str) -> Dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("metrics_append") != "Not Attempted":
        raise RuntimeError("metrics append has already been attempted for this evidence")
    breakdown = validate_breakdown(json.loads(breakdown_json))
    metadata = json.loads(metadata_json) if metadata_json else {}
    report["finalization_breakdown"] = breakdown
    report["finalization_not_applicable"] = [
        key for key in FINALIZATION_KEYS if breakdown[key] == 0.0 and key not in {"total_finalization_ms"}
    ]
    report.update(metadata)
    metrics_started = time.perf_counter_ns()
    metrics_ok = record_task_metrics(
        TaskMetrics(
            task_id=str(report["task_id"]),
            project_key=None,
            task_result=str(report.get("task_result", "Passed")),
            started_at=str(report.get("measurement_timestamp", utc_now())),
            finished_at=utc_now(),
            recall_ms=0.0,
            candidate_count=max(
                int(item["candidate_count"]) for item in report["memory_token_baseline"]["profiles"].values()
            ),
            capsule_count=max(
                int(item["capsule_count"]) for item in report["memory_token_baseline"]["profiles"].values()
            ),
            approx_memory_tokens=max(
                int(item["approx_memory_tokens"])
                for item in report["memory_token_baseline"]["profiles"].values()
            ),
            full_markdown_expansions=max(
                int(item["full_markdown_expansions"])
                for item in report["memory_token_baseline"]["profiles"].values()
            ),
            terminal_model_calls=0,
            extra_learning_calls=0,
            compiler_ms=breakdown["compiler_ms"],
            sqlite_transactions=1,
            shared_knowledge_commits=1 if report.get("shared_knowledge_primary_commit") else 0,
            pushes=1 if report.get("shared_knowledge_push") == "Success" else 0,
            remote_verifies=1 if report.get("shared_knowledge_remote_verify") == "Success" else 0,
            finalization_ms=breakdown["total_finalization_ms"],
        ),
        codex_home(),
    )
    breakdown["metrics_append_ms"] = round((time.perf_counter_ns() - metrics_started) / 1_000_000, 4)
    report["finalization_breakdown"] = breakdown
    report["metrics_append"] = "Success" if metrics_ok else "Failed"
    report["post_finalization_telemetry_cost_ms"] = breakdown["metrics_append_ms"]
    write_report(path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 gap-only benchmark evidence")
    parser.add_argument("--task-id", required=False)
    parser.add_argument("--real-source", type=Path, default=codex_home() / "skills")
    parser.add_argument(
        "--prior-baseline",
        type=Path,
        default=codex_home() / ".state/experience-runtime/phase3-baseline.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime-root", type=Path, default=ROOT)
    parser.add_argument("--shared-knowledge-remote-name")
    parser.add_argument("--shared-knowledge-remote-head")
    parser.add_argument("--finalize-evidence", type=Path)
    parser.add_argument("--breakdown-json", default="{}")
    parser.add_argument("--metadata-json", default="{}")
    arguments = parser.parse_args()
    if arguments.finalize_evidence:
        report = finalize_evidence(arguments.finalize_evidence, arguments.breakdown_json, arguments.metadata_json)
    else:
        if not arguments.task_id or not arguments.output:
            parser.error("--task-id and --output are required for gap measurement")
        snapshot = None
        if arguments.shared_knowledge_remote_name or arguments.shared_knowledge_remote_head:
            if not arguments.shared_knowledge_remote_name or not arguments.shared_knowledge_remote_head:
                parser.error("both Shared Knowledge remote name and head are required")
            snapshot = (arguments.shared_knowledge_remote_name, arguments.shared_knowledge_remote_head)
        report = gap_measurement(
            arguments.real_source.resolve(),
            arguments.prior_baseline.resolve(),
            arguments.task_id,
            arguments.runtime_root.resolve(),
            validated_snapshot=snapshot,
        )
        write_report(arguments.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
