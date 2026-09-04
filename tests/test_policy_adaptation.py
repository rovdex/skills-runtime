from __future__ import annotations

from pathlib import Path

from shared_memory_runtime import (
    RecallCandidate,
    TaskPolicyContext,
    adapt_task_policy,
    compile_task_policy,
    validate_policy_advice,
)
from shared_memory_runtime.markdown import parse_experience_fragment


def _base_policy() -> dict:
    result = compile_task_policy(
        TaskPolicyContext(
            task_intent="Implement a feature",
            available_skills=("backend-development",),
        )
    )
    assert result.policy is not None
    return result.policy


def _candidate(experience_id: str, score: float, *, eligible: bool = True) -> RecallCandidate:
    return RecallCandidate(
        experience_id=experience_id,
        canonical_id=experience_id,
        capsule="capsule",
        score=score,
        effective_experience_verification="verified" if eligible else "candidate",
        remote_verified=eligible,
        exact_anchor_hits=1,
        applicability_passed=eligible,
    )


def test_j3_consumes_authoritative_rank_order_without_reimplementing_ranking() -> None:
    ranked = (_candidate("exp-first", 10.0), _candidate("exp-second", 999.0))
    result = adapt_task_policy(
        _base_policy(),
        ranked,
        {
            "exp-first": {"validation": {"full_suite": True}},
            "exp-second": {"validation": {"full_suite": False}},
        },
        available_skills=("backend-development",),
    )
    assert result.adapted is True
    assert result.source_experience_ids == ("exp-first",)
    assert result.policy["validation"]["full_suite"] is True


def test_j3_equal_ranked_advice_values_are_ambiguous() -> None:
    result = adapt_task_policy(
        _base_policy(),
        (_candidate("exp-a", 4.0), _candidate("exp-b", 4.0)),
        {
            "exp-a": {"validation": {"full_suite": True}},
            "exp-b": {"validation": {"full_suite": False}},
        },
        available_skills=("backend-development",),
    )
    assert result.adapted is False
    assert result.reason_codes == ("advice_winner_ambiguous",)


def test_j3_ineligible_experience_and_advice_none_cannot_adapt() -> None:
    result = adapt_task_policy(
        _base_policy(),
        (_candidate("exp-ineligible", 100.0, eligible=False), _candidate("exp-none", 1.0)),
        {"exp-ineligible": {"validation": {"full_suite": True}}, "exp-none": None},
        available_skills=("backend-development",),
    )
    assert result.adapted is False
    assert result.reason_codes == ("no_eligible_advice",)


def test_j3_kernel_advice_is_rejected() -> None:
    result = adapt_task_policy(
        _base_policy(),
        (_candidate("exp-kernel", 10.0),),
        {"exp-kernel": {"action_gate": False}},
        available_skills=("backend-development",),
    )
    assert result.adapted is False
    assert result.reason_codes == ("kernel_advice_rejected",)
    assert validate_policy_advice({"memory": {"max_capsules": 1}}).valid is True


def test_optional_policy_advice_parses_without_historical_migration(tmp_path: Path) -> None:
    path = tmp_path / ".memory/projects/example-project-3d3378b2/tasks/advice.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        """---
id: 01K00000000000000000000001
type: task
scope: project
project: github.com/example/project
task_id: task-advice
status: completed
confidence: high
importance: 4
title: Advice
summary: Optional advice
experience:
  outcome: NEW
  verification: candidate
  canonical_id: 01K00000000000000000000001
  capsule: Advice capsule
  applies_when: {}
  does_not_apply_when: {}
  policy_advice:
    validation:
      full_suite: true
created_at: 2026-09-04T00:00:00Z
---

# Goal

Advice.

## Experience

### Trigger

Trigger.

### Worked

Worked.

### Does not apply when

None.

### Exceptions

None.

### Evidence

Validation.
""",
        encoding="utf-8",
    )
    record = parse_experience_fragment(path, tmp_path, "hash")
    assert record.policy_advice == {"validation": {"full_suite": True}}
