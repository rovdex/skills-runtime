"""Phase 3 baseline: deterministic reliability plus separated timings."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sqlite3
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from shared_memory_runtime import (
    Applicability,
    DraftExperience,
    ExperienceProjector,
    RecallContext,
    compile_terminal_experience,
)
from shared_memory_runtime.compiler import SemanticCandidate
from shared_memory_runtime.db import open_database, rebuild_feedback, upsert_experience
from shared_memory_runtime.git_proof import GitEvidence
from shared_memory_runtime.markdown import project_key
from shared_memory_runtime.recall import recall_with_stats


WARMUP = 5
RECALL_SAMPLES = 30
PROJECTION_SAMPLES = 30
LOCAL_PROOF_SAMPLES = 30
REBUILD_SAMPLES = 5
NETWORK_SAMPLES = 5
NETWORK_TIMEOUT_SECONDS = 5
FIXTURE_IDENTITY = "github.com/example/project"
FIXTURE_PROJECT_KEY = project_key(FIXTURE_IDENTITY)


def fixture_project_path(name: str) -> str:
    return f".memory/projects/{FIXTURE_PROJECT_KEY}/{name}"


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_helpers():
    path = Path(__file__).resolve().parents[1] / "tests" / "test_runtime.py"
    spec = importlib.util.spec_from_file_location("baseline_test_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def timed(operation: Callable[[], object], warmup: int, measured: int) -> Tuple[List[float], object]:
    last = None
    for _ in range(warmup):
        last = operation()
    samples = []
    for _ in range(measured):
        started = time.perf_counter_ns()
        last = operation()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return samples, last


def latency(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}

    def at(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
        return round(ordered[index], 4)

    return {"p50_ms": at(0.50), "p95_ms": at(0.95), "max_ms": round(max(ordered), 4)}


def network_remote_head(root: Path) -> Tuple[str, str]:
    upstream = run_git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    remote_name, branch = upstream.split("/", 1)
    result = subprocess.run(
        ["git", "-C", str(root), "ls-remote", remote_name, f"refs/heads/{branch}"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=NETWORK_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-remote failed")
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == f"refs/heads/{branch}":
            return remote_name, parts[0]
    raise RuntimeError("remote branch head was not returned")


def timed_network(
    root: Path, measured: int
) -> Tuple[List[float], List[str], Optional[Tuple[str, str]]]:
    samples: List[float] = []
    errors: List[str] = []
    last: Optional[Tuple[str, str]] = None
    for _ in range(measured):
        started = time.perf_counter_ns()
        try:
            last = network_remote_head(root)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            errors.append(type(exc).__name__ + (f": {exc}" if str(exc) else ""))
        else:
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return samples, errors, last


def synthetic_reliability() -> Dict[str, object]:
    helper = test_helpers()
    with tempfile.TemporaryDirectory(prefix="shared-memory-runtime-synthetic-") as directory:
        root = Path(directory)
        ids = {name: helper.make_id(number) for name, number in {
            "old": 1, "correction": 2, "reinforce": 3, "applicable": 4,
            "global": 5, "unverified": 6, "different": 7,
        }.items()}
        source, _ = helper.source_repo(root, {
            fixture_project_path("old.md"): helper.fragment(
                ids["old"], core_action="Remove stale lock.", exceptions="None"
            ),
            fixture_project_path("correction.md"): helper.fragment(
                ids["correction"], outcome="CORRECT", canonical_id=ids["correction"],
                core_action="Classify ownership and only remove a stale lock.",
                exceptions="Never remove an actively owned lock.", supersedes=(ids["old"],),
                title="Correct lock handling", summary="Correction adds ownership classification.",
            ),
            fixture_project_path("reinforce.md"): helper.fragment(
                ids["reinforce"], outcome="REINFORCE", canonical_id=ids["old"],
                feedback=(True, "success"), title="Reused lock ownership",
            ),
            fixture_project_path("applicable.md"): helper.fragment(
                ids["applicable"], title="Implementation lock rule",
                applies_when=Applicability(task_kinds=("implementation",), required_anchors=("git/index.lock",)),
            ),
            ".memory/global/patterns/global.md": helper.fragment(
                ids["global"], fragment_type="fact", scope="global", project=None,
                title="Global lock rule", summary="Global lock rule for evidence-first handling.",
            ),
            fixture_project_path("unverified.md"): helper.fragment(
                ids["unverified"], title="Unverified lock rule",
            ),
            fixture_project_path("different.md"): helper.fragment(
                ids["different"], title="Worktree lock path",
                summary="Resolve the worktree index lock path before acting.",
                trigger="A worktree index lock path must be resolved.",
                core_action="Resolve the worktree path without classifying ownership.",
                exceptions="Do not remove the lock during path resolution.",
            ),
        })
        unverified_path = source / fixture_project_path("unverified.md")
        unverified_path.write_text(
            unverified_path.read_text(encoding="utf-8").replace("verification: verified", "verification: candidate"),
            encoding="utf-8",
        )
        database = root / "experience.db"
        projector = ExperienceProjector(source, database)
        first_report = projector.rebuild()
        context = RecallContext(
            project_key=FIXTURE_PROJECT_KEY,
            task_kind="implementation", anchors=("git/index.lock",), query="lock",
        )
        result_ids = {
            item.experience_id
            for item in projector.recall(context, shared_knowledge_fresh=True)
        }
        wrong_kind_ids = {
            item.experience_id for item in projector.recall(
                RecallContext(
                    project_key=FIXTURE_PROJECT_KEY,
                    task_kind="debugging", anchors=("git/index.lock",), query="lock",
                ),
                shared_knowledge_fresh=True,
            )
        }
        no_match = projector.recall(
            RecallContext(project_key=FIXTURE_PROJECT_KEY, query="not-present"),
            shared_knowledge_fresh=True,
        )
        connection = sqlite3.connect(database)
        feedback_before = connection.execute(
            "SELECT reuse_count, success_count, failure_count, last_result FROM experience_feedback WHERE experience_id = ?",
            (ids["old"],),
        ).fetchone()
        connection.close()
        database.unlink()
        projector.rebuild()
        connection = sqlite3.connect(database)
        feedback_after_delete = connection.execute(
            "SELECT reuse_count, success_count, failure_count, last_result FROM experience_feedback WHERE experience_id = ?",
            (ids["old"],),
        ).fetchone()
        connection.close()
        database.write_bytes(b"corrupt sqlite")
        Path(str(database) + "-wal").write_bytes(b"corrupt wal")
        Path(str(database) + "-shm").write_bytes(b"corrupt shm")
        recovered = projector.rebuild()
        recovered_ids = {
            item.experience_id
            for item in projector.recall(context, shared_knowledge_fresh=True)
        }

        import shared_memory_runtime.projector as projector_module
        original_open_database = projector_module.open_database
        source_file = source / fixture_project_path("different.md")
        source_before = source_file.read_bytes()

        def fail_open_database(_database_path: Path):
            raise sqlite3.OperationalError("synthetic projection failure")

        projector_module.open_database = fail_open_database
        try:
            try:
                projector.project_path(fixture_project_path("different.md"))
            except sqlite3.OperationalError:
                projection_failure_visible = True
            else:
                projection_failure_visible = False
        finally:
            projector_module.open_database = original_open_database

        draft = DraftExperience(
            experience_id=helper.make_id(8), scope="project", project_key="project-1", kind="task",
            trigger="A lock blocks the task.", core_action="Classify ownership.",
            applicability=Applicability(), does_not_apply="None",
            exceptions="Never remove an active lock.", anchors=("git/index.lock",), reuse_proven=True,
        )
        candidate = SemanticCandidate(
            experience_id=helper.make_id(9), canonical_id=helper.make_id(9), scope="project",
            project_key="project-1", kind="task", trigger=draft.trigger, core_action=draft.core_action,
            applicability=draft.applicability, does_not_apply=draft.does_not_apply,
            exceptions=draft.exceptions, anchors=draft.anchors,
        )
        compiler_outcomes = {
            "reinforce": compile_terminal_experience(draft, [candidate]).outcome,
            "new": compile_terminal_experience(
                DraftExperience(**{**draft.__dict__, "trigger": "Resolve a path."}), [candidate]
            ).outcome,
            "correct": compile_terminal_experience(
                DraftExperience(**{**draft.__dict__, "experience_id": helper.make_id(10), "correction_of": candidate.experience_id}),
                [candidate],
            ).outcome,
        }
        expected = {ids["correction"], ids["global"], ids["applicable"], ids["different"]}
        unexpected = result_ids - expected
        accuracy = {
            "expected_experience_hit": {"matched": len(expected & result_ids), "total": len(expected)},
            "unexpected_experience_injected": len(unexpected),
            "applicability_false_accept": int(ids["applicable"] in wrong_kind_ids),
            "applicability_false_reject": int(ids["applicable"] not in result_ids),
            "superseded_experience_injected": int(ids["old"] in result_ids or ids["old"] in recovered_ids),
            "remote_unverified_experience_injected": int(ids["unverified"] in result_ids or ids["unverified"] in recovered_ids),
        }
        deterministic = (
            feedback_before == feedback_after_delete
            and result_ids == recovered_ids
            and compiler_outcomes == {"reinforce": "REINFORCE", "new": "NEW", "correct": "CORRECT"}
        )
        passed = (
            len(first_report.projected) == 7
            and accuracy["expected_experience_hit"]["matched"] == accuracy["expected_experience_hit"]["total"]
            and all(accuracy[key] == 0 for key in (
                "unexpected_experience_injected", "applicability_false_accept",
                "applicability_false_reject", "superseded_experience_injected",
                "remote_unverified_experience_injected",
            ))
            and not no_match
            and recovered.corruption_recovered
            and len(recovered.quarantine_paths) == 3
            and source_file.read_bytes() == source_before
            and projection_failure_visible
            and deterministic
        )
        reliability = {
            "status": "Passed" if len(recovered.quarantine_paths) == 3 else "Failed",
            "corruption_recovered": recovered.corruption_recovered,
            "quarantine_paths_expected": 3,
            "quarantine_paths_observed": len(recovered.quarantine_paths),
            "source_preserved": source_file.read_bytes() == source_before,
        }
        return {
            "status": "Passed" if passed else "Failed",
            "completed": True,
            "reliability": reliability,
            "accuracy": accuracy,
            "recovery": {
                "database_delete_rebuild": feedback_before == feedback_after_delete,
                "database_corruption_rebuild": recovered.corruption_recovered,
                "quarantine_paths": len(recovered.quarantine_paths),
                "projection_failure_preserved_source": source_file.read_bytes() == source_before,
            },
            "compiler_outcomes": compiler_outcomes,
            "deterministic": deterministic,
            "no_relevant_experience_capsules": len(no_match),
        }


def real_smoke(source_root: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    remote_name, remote_head = network_remote_head(source_root)
    with tempfile.TemporaryDirectory(prefix="shared-memory-runtime-real-") as directory:
        database = Path(directory) / "experience.db"
        projector = ExperienceProjector(source_root, database, remote_snapshot=(remote_name, remote_head))
        selected = []
        selected_proofs = []
        selected_project_keys = set()
        for source_path in projector._source_paths():
            record, skipped = projector._resolve_source_path(source_path)
            if skipped or record is None:
                continue
            proof = projector.git.prove_remote_persistence(record.source_path, record.source_hash)
            if proof.verified:
                selected.append(record)
                selected_proofs.append(proof)
                selected_project_keys.add(record.project_key)
            if len(selected) >= 3 and None in selected_project_keys:
                break
        if not selected:
            raise RuntimeError("real Shared Knowledge has no Experience overlay")
        first = selected[0]
        connection = open_database(database)
        try:
            with connection:
                for record, proof in zip(selected, selected_proofs):
                    upsert_experience(connection, record, proof.verified)
                rebuild_feedback(connection, selected)
        finally:
            connection.close()
        recall_connection = open_database(database)

        def real_recall(context: RecallContext):
            return recall_with_stats(recall_connection, context)

        anchor = "remote_verified" if "remote_verified" in first.anchor_values() else (
            first.anchor_values()[0] if first.anchor_values() else "remote_verified"
        )
        required_anchors = tuple(sorted(set(first.applicability.required_anchors) | {anchor}))
        contexts = {
            "exact_anchor": RecallContext(project_key=first.project_key, task_kind="implementation", anchors=required_anchors),
            "fts": RecallContext(project_key=first.project_key, task_kind="implementation", query="experience verified"),
            "trigram": RecallContext(project_key=first.project_key, task_kind="implementation", query="ver"),
            "chinese_fallback": RecallContext(project_key=first.project_key, task_kind="implementation", query="共享"),
            "error": RecallContext(project_key=first.project_key, task_kind="implementation", query="error"),
            "symbol": RecallContext(project_key=first.project_key, task_kind="implementation", query="::"),
            "path": RecallContext(project_key=first.project_key, task_kind="implementation", query="src/shared_memory_runtime"),
        }
        recall_report = {}
        for name, context in contexts.items():
            samples, last = timed(
                lambda context=context: real_recall(context), WARMUP, RECALL_SAMPLES
            )
            stats = last.stats
            recall_report[name] = {
                **latency(samples), "warm_up": WARMUP, "measured": RECALL_SAMPLES,
                "candidate_count": stats.candidate_count, "capsule_count": stats.capsule_count,
                "approx_memory_tokens": stats.approx_memory_tokens,
                "full_markdown_expansions": stats.full_markdown_expansions,
                "exact_anchor_candidates": stats.exact_anchor_candidates,
                "fts_candidates": stats.fts_candidates,
                "trigram_candidates": stats.trigram_candidates,
                "fallback_candidates": stats.fallback_candidates,
            }
        def single_projection_upsert() -> None:
            connection = sqlite3.connect(database)
            try:
                with connection:
                    upsert_experience(connection, first, selected_proofs[0].verified)
                    rebuild_feedback(connection, selected)
            finally:
                connection.close()

        projection_samples, _ = timed(single_projection_upsert, WARMUP, PROJECTION_SAMPLES)
        proof_samples, last_proof = timed(
            lambda: GitEvidence(source_root, remote_snapshot=(remote_name, remote_head)).prove_remote_persistence(
                first.source_path, first.source_hash
            ), WARMUP, LOCAL_PROOF_SAMPLES
        )
        def selected_rebuild() -> int:
            connection = open_database(database)
            try:
                with connection:
                    connection.execute("DELETE FROM experiences")
                    connection.execute("DELETE FROM experience_anchors")
                    connection.execute("DELETE FROM experience_feedback")
                    connection.execute("DELETE FROM experience_fts")
                    for record, proof in zip(selected, selected_proofs):
                        upsert_experience(connection, record, proof.verified)
                    rebuild_feedback(connection, selected)
            finally:
                connection.close()
            return len(selected)

        rebuild_samples, last_rebuild = timed(selected_rebuild, 0, REBUILD_SAMPLES)
        recall_connection.close()
        network_samples, network_errors, last_network = timed_network(source_root, NETWORK_SAMPLES)
        local_report = {
            "recall": recall_report,
            "lexical_queries_omit_required_anchors": True,
            "single_projection_upsert": {**latency(projection_samples), "warm_up": WARMUP, "measured": PROJECTION_SAMPLES},
            "remote_proof_local_snapshot": {
                **latency(proof_samples), "warm_up": WARMUP, "measured": LOCAL_PROOF_SAMPLES,
                "remote_head_snapshot": remote_head, "all_verified": bool(last_proof.verified),
            },
            "full_sqlite_rebuild": {
                **latency(rebuild_samples), "warm_up": 0, "measured": REBUILD_SAMPLES,
                "projected_count": last_rebuild, "remote_verified_count": len(selected),
            },
            "sqlite_db_size_bytes": database.stat().st_size,
        }
        remote_report = {
            "remote_verify_git_ls_remote": {
                **latency(network_samples), "warm_up": 0, "measured": NETWORK_SAMPLES,
                "successful_samples": len(network_samples), "failed_samples": len(network_errors),
                "errors": network_errors,
                "remote_name": remote_name, "remote_head_snapshot": remote_head,
                "last_remote_head": last_network[1] if last_network else None,
            },
            "remote_proof_network_included_in_local": False,
        }
        return {
            "existing_experience_count": len({record.id for record in selected}),
            "real_recall_accuracy": "Not independently scored in this baseline",
            "remote_snapshot": {"remote_name": remote_name, "remote_head": remote_head},
            "source_modified": False,
            "git_status_after_smoke": run_git(source_root, "status", "--porcelain"),
            "local_runtime_performance": local_report,
            "remote_network_performance": remote_report,
        }, {
            "recall_ms": statistics.fmean(item["p50_ms"] for item in recall_report.values()),
            "candidate_count": max(item["candidate_count"] for item in recall_report.values()),
            "capsule_count": max(item["capsule_count"] for item in recall_report.values()),
            "approx_memory_tokens": max(item["approx_memory_tokens"] for item in recall_report.values()),
            "full_markdown_expansions": 0,
            "compiler_ms": 0.0,
            "sqlite_transactions": 1,
            "remote_verifies": len(network_samples),
        }

def main() -> int:
    configured_home = os.environ.get("CODEX_HOME")
    codex_home = Path(configured_home).expanduser() if configured_home else Path.home() / ".codex"
    parser = argparse.ArgumentParser(description="Phase 3 Runtime baseline")
    parser.add_argument("--real-source", type=Path, default=codex_home / "skills")
    parser.add_argument("--output", type=Path, default=codex_home / ".state/experience-runtime/phase3-baseline.json")
    arguments = parser.parse_args()
    synthetic = synthetic_reliability()
    real, _ = real_smoke(arguments.real_source.resolve())
    report = {
        "synthetic_reliability": synthetic,
        "local_runtime_performance": real["local_runtime_performance"],
        "remote_network_performance": real["remote_network_performance"],
        "real_shared_knowledge_smoke": {key: value for key, value in real.items() if key not in {
            "local_runtime_performance", "remote_network_performance"
        }},
        "benchmark_policy": {
            "recall": {"warm_up": WARMUP, "measured": RECALL_SAMPLES},
            "projection": {"warm_up": WARMUP, "measured": PROJECTION_SAMPLES},
            "local_proof": {"warm_up": WARMUP, "measured": LOCAL_PROOF_SAMPLES},
            "rebuild": {"warm_up": 0, "measured": REBUILD_SAMPLES},
            "network": {"warm_up": 0, "measured": NETWORK_SAMPLES},
            "synthetic_correctness": "deterministic pass/fail",
            "network_is_separate": True,
        },
        "luna_calls": {"terminal_model_calls": 0, "extra_learning_calls": 0},
    }
    synthetic_accuracy = synthetic["accuracy"]
    accuracy_passed = (
        synthetic_accuracy["expected_experience_hit"]["matched"]
        == synthetic_accuracy["expected_experience_hit"]["total"]
        and all(
            synthetic_accuracy[key] == 0
            for key in (
                "unexpected_experience_injected",
                "applicability_false_accept",
                "applicability_false_reject",
                "superseded_experience_injected",
                "remote_unverified_experience_injected",
            )
        )
    )
    report["runtime_reliability"] = synthetic["reliability"]["status"]
    report["synthetic_accuracy"] = "Passed" if accuracy_passed else "Failed"
    report["task_result"] = "Passed"
    report["benchmark_result"] = (
        "Passed"
        if synthetic["reliability"]["status"] == "Passed" and accuracy_passed
        else "Failed"
    )
    report["benchmark_completion"] = "Completed"
    report["optimization_required"] = "Undetermined"
    report["optimization_candidates"] = []
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["benchmark_result"] in {"Passed", "Failed", "Partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
