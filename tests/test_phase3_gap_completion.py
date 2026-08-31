from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase3_gap_completion", ROOT / "benchmarks" / "phase3_gap_completion.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_named_profiles_preserve_existing_recall_limits() -> None:
    contexts = MODULE.profile_contexts("example-project", "implementation", ("anchor",))

    assert tuple(contexts) == ("Normal", "Complex", "Debugging")
    assert contexts["Normal"].task_class == "normal"
    assert contexts["Complex"].task_class == "complex"
    assert contexts["Debugging"].task_class == "debugging"
    assert contexts["Normal"].limit == 4
    assert contexts["Complex"].limit == 5
    assert contexts["Debugging"].limit == 5
    assert all(context.expand is False for context in contexts.values())


def test_breakdown_has_all_required_keys_and_rejects_negative_values() -> None:
    breakdown = MODULE.validate_breakdown({"total_finalization_ms": 12.5})

    assert set(breakdown) == set(MODULE.FINALIZATION_KEYS)
    assert breakdown["total_finalization_ms"] == 12.5
    with pytest.raises(ValueError):
        MODULE.validate_breakdown({"compiler_ms": -1})


def test_prior_baseline_audit_is_read_only(tmp_path: Path) -> None:
    baseline = tmp_path / "phase3-baseline.json"
    baseline.write_text('{"local_runtime_performance": {}}\n', encoding="utf-8")
    before = hashlib.sha256(baseline.read_bytes()).hexdigest()

    audit = MODULE.audit_prior_baseline(baseline)

    assert audit["sha256"] == before
    assert audit["unchanged_during_audit"] is True
    assert set(audit["baseline_gaps"]) == {
        "named_memory_profiles",
        "detailed_finalization_breakdown",
    }
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == before
