"""Deterministic, provider-derived Task Policy compilation for J2.

This module is intentionally separate from ``compiler.py``.  The latter is
the Terminal Experience Compiler; this module compiles only the bounded,
non-Kernel recommendation consumed by the JIT roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


POLICY_REGISTRY_VERSION = "j2-v1"

TASK_CLASSES = (
    "bug_fix",
    "feature",
    "api",
    "documentation",
    "code_review",
    "research",
    "maintenance",
    "benchmark",
    "recovery",
    "general",
)

POLICY_ENUMS = {
    "task_class": TASK_CLASSES,
    # Fast/Standard/Deep are existing terminology in the authoritative
    # development Skills.  The lowercase wire values are canonicalized here.
    "planning.depth": ("fast", "standard", "deep"),
    "memory.scope": ("global", "project"),
    "exploration.strategy": ("fast", "standard", "deep"),
}

NUMERIC_BOUNDS = {
    "memory.max_capsules": (0, 8),
    "exploration.max_read_lines": (50, 500),
    "exploration.max_output_kb": (1, 1024),
}

KERNEL_FIELDS = frozenset(
    {
        "task_id",
        "runtime_state",
        "action_gate",
        "git_safety",
        "single_writer",
        "primary",
        "sedimentation",
        "remote_verify",
        "receipt",
    }
)

POLICY_SHAPE = {
    "task_class": None,
    "planning": {"depth": None},
    "capabilities": {"skills": None},
    "memory": {
        "scope": None,
        "max_capsules": None,
        "full_markdown": None,
    },
    "exploration": {
        "strategy": None,
        "max_read_lines": None,
        "max_output_kb": None,
    },
    "execution": {
        "shell_batching": None,
        "wsl_batching": None,
        "parallel_exploration": None,
    },
    "validation": {
        "focused_tests": None,
        "full_suite": None,
    },
}

# These values are only used where no authoritative provider exists.  They
# are not claims about the current Harness and never affect execution in J2.
SHADOW_BASELINE = {
    "planning.depth": "standard",
    "memory.scope": "project",
    "memory.max_capsules": 1,
    "memory.full_markdown": False,
    "exploration.strategy": "standard",
    "exploration.max_read_lines": 200,
    "exploration.max_output_kb": 64,
    "execution.shell_batching": False,
    "execution.wsl_batching": False,
    "execution.parallel_exploration": False,
    "validation.focused_tests": True,
    "validation.full_suite": False,
}


@dataclass(frozen=True)
class PolicyProvider:
    """One authoritative/default provider candidate for a policy field."""

    source: str
    value: object
    precedence: int = 0


@dataclass(frozen=True)
class TaskPolicyContext:
    task_intent: str = ""
    formal_task_metadata: Mapping[str, object] = field(default_factory=dict)
    available_skills: Tuple[str, ...] = ()
    selected_skills: Tuple[str, ...] = ()
    project_metadata: Mapping[str, object] = field(default_factory=dict)
    environment_metadata: Mapping[str, object] = field(default_factory=dict)
    existing_providers: Mapping[str, object] = field(default_factory=dict)
    skill_providers: Mapping[str, object] = field(default_factory=dict)
    # Accepted for forward compatibility only.  J2 never reads it to adapt a
    # policy; J3 owns Experience -> Policy adaptation.
    experience_metadata: Optional[object] = None


@dataclass(frozen=True)
class PolicyValidation:
    valid: bool
    reason_codes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyCompileResult:
    policy: Optional[Mapping[str, object]]
    valid: bool
    fallback: bool
    task_class: str
    reason_codes: Tuple[str, ...] = ()
    provenance: Mapping[str, str] = field(default_factory=dict)
    registry_version: str = POLICY_REGISTRY_VERSION
    experience_ignored: bool = False


def _provider_tuple(value: object) -> Tuple[PolicyProvider, ...]:
    if value is None:
        return ()
    if isinstance(value, PolicyProvider):
        return (value,)
    if isinstance(value, (list, tuple)):
        providers = tuple(value)
        if not all(isinstance(item, PolicyProvider) for item in providers):
            raise TypeError("provider candidates must be PolicyProvider values")
        return providers
    raise TypeError("provider candidates must be a PolicyProvider or sequence")


def _resolve_provider(
    path: str,
    existing: Mapping[str, object],
    skill: Mapping[str, object],
) -> Tuple[object, str, Optional[str]]:
    """Resolve existing > Skill > shadow, preserving ambiguity as failure."""

    for mapping, expected_source in ((existing, "existing_default"), (skill, "skill_default")):
        candidates = _provider_tuple(mapping.get(path))
        if not candidates:
            continue
        if any(item.source != expected_source for item in candidates):
            return None, "", "provider_source_invalid"
        highest = max(item.precedence for item in candidates)
        winners = [item for item in candidates if item.precedence == highest]
        values = {repr(item.value) for item in winners}
        if len(values) != 1:
            return None, "", "provider_ambiguity"
        return winners[0].value, expected_source, None
    return SHADOW_BASELINE[path], "shadow_baseline", None


def _set_path(target: Dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    cursor: Dict[str, object] = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _get_path(value: Mapping[str, object], path: str) -> object:
    cursor: object = value
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _flatten_paths(value: Mapping[str, object], prefix: str = "") -> Iterable[str]:
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        yield path
        if isinstance(child, Mapping):
            yield from _flatten_paths(child, path)


def _class_from_skills(skills: Sequence[str]) -> Optional[str]:
    names = {str(skill).strip() for skill in skills}
    for task_class, candidates in (
        ("bug_fix", ("bug-debugging",)),
        ("code_review", ("code-review",)),
        ("api", ("api-development",)),
        ("feature", ("backend-development",)),
        ("research", ("prototype-analysis", "deep-research-work:deep-research")),
        ("recovery", ("skill-sync",)),
        ("maintenance", ("development-tooling", "skill-development")),
    ):
        if names.intersection(candidates):
            return task_class
    return None


def _class_from_intent(intent: str) -> Optional[str]:
    lowered = intent.casefold()
    signals = (
        ("bug_fix", ("bug", "error", "exception", "regression", "broken", "fix")),
        ("api", ("api", "endpoint", "rpc", "dto", "serializer")),
        ("code_review", ("code review", "review", "diff", "pull request", "pr")),
        ("documentation", ("documentation", "docs", "readme")),
        ("research", ("research", "prototype", "investigate", "analysis")),
        ("benchmark", ("benchmark", "performance", "latency")),
        ("recovery", ("recovery", "resume", "receipt", "finalization blocker")),
        ("maintenance", ("maintenance", "tooling", "dependency", "sync")),
        ("feature", ("feature", "implement", "build", "add")),
    )
    for task_class, values in signals:
        if any(value in lowered for value in values):
            return task_class
    return None


def classify_task(context: TaskPolicyContext) -> Tuple[str, str]:
    """Return ``(task_class, provenance)`` without consulting Experience."""

    metadata_value = context.formal_task_metadata.get("task_class")
    if isinstance(metadata_value, str) and metadata_value in TASK_CLASSES:
        return metadata_value, "existing_default"
    skill_class = _class_from_skills(context.selected_skills)
    if skill_class:
        return skill_class, "skill_default"
    intent_class = _class_from_intent(context.task_intent)
    if intent_class:
        return intent_class, "task_intent"
    return "general", "shadow_baseline"


def _selected_skills(task_class: str, context: TaskPolicyContext) -> Tuple[str, ...]:
    routes = {
        "bug_fix": ("bug-debugging",),
        "feature": ("backend-development",),
        "api": ("api-development", "backend-development"),
        "documentation": (),
        "code_review": ("code-review",),
        "research": ("prototype-analysis", "deep-research-work:deep-research"),
        "maintenance": ("development-tooling",),
        "benchmark": ("development-tooling",),
        "recovery": ("skill-sync", "development-tooling"),
        "general": (),
    }
    available = set(context.available_skills)
    selected = set(routes.get(task_class, ())).intersection(available)
    lowered = context.task_intent.casefold()
    extra_routes = {
        "database-development": ("database", "schema", "migration", "index"),
        "security-development": ("security", "authorization", "permission", "tenant", "secret"),
        "skill-development": ("skill",),
        "skill-sync": ("shared knowledge", "sync"),
    }
    for skill, signals in extra_routes.items():
        if skill in available and any(signal in lowered for signal in signals):
            selected.add(skill)
    return tuple(sorted(selected))


def _base_policy(task_class: str, context: TaskPolicyContext) -> Dict[str, object]:
    policy: Dict[str, object] = {
        "task_class": task_class,
        "planning": {"depth": SHADOW_BASELINE["planning.depth"]},
        "capabilities": {"skills": list(_selected_skills(task_class, context))},
        "memory": {
            "scope": SHADOW_BASELINE["memory.scope"],
            "max_capsules": SHADOW_BASELINE["memory.max_capsules"],
            "full_markdown": SHADOW_BASELINE["memory.full_markdown"],
        },
        "exploration": {
            "strategy": SHADOW_BASELINE["exploration.strategy"],
            "max_read_lines": SHADOW_BASELINE["exploration.max_read_lines"],
            "max_output_kb": SHADOW_BASELINE["exploration.max_output_kb"],
        },
        "execution": {
            "shell_batching": SHADOW_BASELINE["execution.shell_batching"],
            "wsl_batching": SHADOW_BASELINE["execution.wsl_batching"],
            "parallel_exploration": SHADOW_BASELINE["execution.parallel_exploration"],
        },
        "validation": {
            "focused_tests": SHADOW_BASELINE["validation.focused_tests"],
            "full_suite": SHADOW_BASELINE["validation.full_suite"],
        },
    }
    return policy


def validate_task_policy(
    policy: Mapping[str, object],
    *,
    available_skills: Sequence[str] = (),
) -> PolicyValidation:
    """Validate only the J2 whitelist in the frozen order."""

    if not isinstance(policy, Mapping):
        return PolicyValidation(False, ("policy_not_mapping",))
    allowed_paths = set(_flatten_paths(POLICY_SHAPE))
    for path in _flatten_paths(policy):
        if path not in allowed_paths:
            if path in KERNEL_FIELDS or path.split(".")[-1] in KERNEL_FIELDS:
                return PolicyValidation(False, ("kernel_field_rejected",))
            return PolicyValidation(False, ("unknown_policy_field",))

    expected_types = {
        "task_class": str,
        "planning.depth": str,
        "capabilities.skills": list,
        "memory.scope": str,
        "memory.max_capsules": int,
        "memory.full_markdown": bool,
        "exploration.strategy": str,
        "exploration.max_read_lines": int,
        "exploration.max_output_kb": int,
        "execution.shell_batching": bool,
        "execution.wsl_batching": bool,
        "execution.parallel_exploration": bool,
        "validation.focused_tests": bool,
        "validation.full_suite": bool,
    }
    for path, expected in expected_types.items():
        value = _get_path(policy, path)
        if expected is int:
            valid_type = isinstance(value, int) and not isinstance(value, bool)
        elif expected is list:
            valid_type = isinstance(value, list) and all(isinstance(item, str) for item in value)
        else:
            valid_type = isinstance(value, expected)
        if not valid_type:
            return PolicyValidation(False, (f"invalid_type:{path}",))

    for path, values in POLICY_ENUMS.items():
        if _get_path(policy, path) not in values:
            return PolicyValidation(False, (f"invalid_enum:{path}",))

    for path, (minimum, maximum) in NUMERIC_BOUNDS.items():
        value = _get_path(policy, path)
        if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
            return PolicyValidation(False, (f"invalid_bounds:{path}",))

    skills = _get_path(policy, "capabilities.skills")
    if len(set(skills)) != len(skills):
        return PolicyValidation(False, ("duplicate_skill",))
    unavailable = sorted(set(skills) - set(available_skills))
    if unavailable:
        return PolicyValidation(False, ("skill_missing",))
    return PolicyValidation(True)


def compile_task_policy(context: TaskPolicyContext) -> PolicyCompileResult:
    """Compile a bounded shadow recommendation without Experience adaptation."""

    task_class, class_provenance = classify_task(context)
    policy = _base_policy(task_class, context)
    provenance: Dict[str, str] = {"task_class": class_provenance, "capabilities.skills": "task_routing"}
    reason_codes = []

    for path in (
        "planning.depth",
        "memory.scope",
        "memory.max_capsules",
        "memory.full_markdown",
        "exploration.strategy",
        "exploration.max_read_lines",
        "exploration.max_output_kb",
        "execution.shell_batching",
        "execution.wsl_batching",
        "execution.parallel_exploration",
        "validation.focused_tests",
        "validation.full_suite",
    ):
        try:
            value, source, error = _resolve_provider(
                path, context.existing_providers, context.skill_providers
            )
        except TypeError:
            return PolicyCompileResult(
                None,
                False,
                True,
                task_class,
                ("provider_invalid",),
                provenance,
                experience_ignored=context.experience_metadata is not None,
            )
        if error:
            return PolicyCompileResult(
                None,
                False,
                True,
                task_class,
                (error,),
                provenance,
                experience_ignored=context.experience_metadata is not None,
            )
        _set_path(policy, path, value)
        provenance[path] = source

    validation = validate_task_policy(policy, available_skills=context.available_skills)
    if not validation.valid:
        reason_codes.extend(validation.reason_codes)
        return PolicyCompileResult(
            None,
            False,
            True,
            task_class,
            tuple(reason_codes),
            provenance,
            experience_ignored=context.experience_metadata is not None,
        )
    if context.experience_metadata is not None:
        reason_codes.append("experience_ignored_for_j2")
    return PolicyCompileResult(
        policy=policy,
        valid=True,
        fallback=False,
        task_class=task_class,
        reason_codes=tuple(reason_codes),
        provenance=provenance,
        experience_ignored=context.experience_metadata is not None,
    )


__all__ = [
    "KERNEL_FIELDS",
    "NUMERIC_BOUNDS",
    "POLICY_ENUMS",
    "POLICY_REGISTRY_VERSION",
    "PolicyCompileResult",
    "PolicyProvider",
    "PolicyValidation",
    "SHADOW_BASELINE",
    "TASK_CLASSES",
    "TaskPolicyContext",
    "classify_task",
    "compile_task_policy",
    "validate_task_policy",
]
