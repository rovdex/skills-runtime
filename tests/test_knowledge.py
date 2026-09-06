from __future__ import annotations

from pathlib import Path

from shared_memory_runtime import (
    SkillAdvice,
    SkillRecallContext,
    TaskPolicyContext,
    TaskMetrics,
    adapt_task_policy,
    compile_task_policy,
    knowledge_root,
    prepare_formal_task_jit,
    recall_skills,
    skill_evolution_check,
    validate_knowledge_tree,
)


def _root() -> Path:
    return knowledge_root()


def _base_policy() -> dict:
    result = compile_task_policy(
        TaskPolicyContext(
            task_intent="Implement a feature",
            available_skills=("backend-development",),
        )
    )
    assert result.policy is not None
    return result.policy


def test_portable_index_and_git_rebuildable_evolution() -> None:
    root = _root()
    findings = validate_knowledge_tree(root, require_git_tracked=True)
    assert findings == ("skills=2", "index=direct-readable", "evidence=git-rebuildable")
    for skill_id in (
        "general.evidence-bounded-verification",
        "software.runtime-contract-parity",
    ):
        result = skill_evolution_check(root, skill_id)
        assert result.current_status == "candidate"
        assert result.decision == "CANDIDATE_SKILL"
        assert result.candidate_support_count == 2
        assert result.successful_reuse_count == 0


def test_capsule_first_and_explicit_full_skill_expansion() -> None:
    root = _root()
    context = SkillRecallContext(
        task="formal verification with durable evidence",
        domain="general",
        task_type="formal-task",
        anchors=("durable evidence",),
    )
    capsule_only = recall_skills(root, context)
    assert capsule_only.stats.capsule_count == 1
    assert capsule_only.stats.full_skill_count == 0
    assert capsule_only.stats.experience_expansion_count == 0
    assert capsule_only.candidates[0].skill_id == "general.evidence-bounded-verification"

    expanded = recall_skills(
        root,
        context,
        full_skill_ids=("general.evidence-bounded-verification",),
    )
    assert expanded.stats.full_skill_count == 1
    assert expanded.full_skills[0].lstrip().startswith("# Goal")


def test_general_and_software_skills_can_compose() -> None:
    result = recall_skills(
        _root(),
        SkillRecallContext(
            task="formal Runtime parity verification with durable evidence",
            task_type="formal-task",
            anchors=("durable evidence", "Runtime Compatibility"),
        ),
    )
    assert [item.skill_id for item in result.candidates] == [
        "software.runtime-contract-parity",
        "general.evidence-bounded-verification",
    ]


def test_no_applicable_skill_uses_experience_fallback() -> None:
    result = recall_skills(
        _root(),
        SkillRecallContext(task="unrelated task", domain="unrelated", task_type="other"),
    )
    assert result.candidates == ()
    assert result.stats.experience_expansion_count == 2
    assert all(item.verification == "verified" for item in result.experiences)


def test_candidate_guidance_is_not_j3_advice_but_verified_skill_is() -> None:
    result = adapt_task_policy(
        _base_policy(),
        (),
        {},
        available_skills=("backend-development",),
        verified_skill_advice=(
            SkillAdvice(
                skill_id="general.evidence-bounded-verification",
                status="candidate",
                fields={"validation": {"full_suite": True}},
            ),
        ),
    )
    assert result.adapted is False
    assert result.source_skill_ids == ()
    assert result.reason_codes == ("no_eligible_advice",)

    verified = adapt_task_policy(
        _base_policy(),
        (),
        {},
        available_skills=("backend-development",),
        verified_skill_advice=(
            SkillAdvice(
                skill_id="software.runtime-contract-parity",
                status="verified",
                fields={"validation": {"full_suite": True}},
            ),
        ),
    )
    assert verified.adapted is True
    assert verified.source_skill_ids == ("software.runtime-contract-parity",)
    assert verified.policy["validation"]["full_suite"] is True


def test_formal_jit_exposes_verified_skill_knowledge_adaptation(tmp_path: Path) -> None:
    result = prepare_formal_task_jit(
        "66836386-06c7-4ad9-b1ca-33becab9feac",
        context=TaskPolicyContext(
            task_intent="Implement Runtime parity validation",
            available_skills=("skill-development", "skill-sync"),
        ),
        verified_skill_advice=(
            SkillAdvice(
                skill_id="software.runtime-contract-parity",
                status="verified",
                fields={"validation": {"full_suite": True}},
            ),
        ),
        codex_home=tmp_path,
    )
    assert result.evidence["j3"]["knowledge_adaptation"] == "Verified Skill"
    assert result.evidence["j3"]["verified_skill_policy_sources"] == [
        "software.runtime-contract-parity"
    ]


def test_metrics_keep_old_fields_and_add_portable_knowledge_fields() -> None:
    metrics = TaskMetrics(
        task_id="knowledge-test",
        project_key=None,
        task_result="Passed",
        started_at="2026-09-06T00:00:00Z",
        finished_at="2026-09-06T00:00:01Z",
        capsule_count=2,
        full_skill_count=1,
        experience_expansion_count=0,
        approx_injected_knowledge_tokens=240,
    )
    mapping = metrics.as_mapping()
    assert mapping["capsule_count"] == 2
    assert mapping["full_skill_count"] == 1
    assert mapping["experience_expansion_count"] == 0
    assert mapping["approx_injected_knowledge_tokens"] == 240
    assert mapping["full_markdown_expansions"] == 0
