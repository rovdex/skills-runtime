"""Parse and validate authoritative Shared Memory Markdown Fragments."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import yaml

from .models import Applicability, ExperienceRecord, FeedbackEvent


ALLOWED_FRAGMENT_TYPES = {"fact", "decision", "case", "task", "validation", "handoff"}
FORMAL_PRIMARY_TYPES = {"task", "case", "handoff"}
ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$", re.MULTILINE)


class FragmentValidationError(ValueError):
    """Raised when an Experience-bearing Primary Fragment is not valid."""

    def __init__(self, message: str, *, reason_code: Optional[str] = None):
        super().__init__(message)
        self.reason_code = reason_code


def normalize_project_identity(identity: str) -> str:
    value = identity.strip()
    if not value:
        raise FragmentValidationError("project identity is empty")
    scheme_match = re.match(r"^(https?|ssh)://", value, flags=re.IGNORECASE)
    scheme = scheme_match.group(1).casefold() if scheme_match else ""
    value = re.sub(r"^(https?|ssh)://", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^git@", "", value, flags=re.IGNORECASE)
    value = value.strip("/")
    host, separator, path = value.partition("/")
    if not separator or not path:
        raise FragmentValidationError("project identity must contain host and repository")
    if ":" in host:
        host_name, port = host.rsplit(":", 1)
        if port.isdigit():
            default_port = (scheme == "http" and port == "80") or (
                scheme == "https" and port == "443"
            ) or (scheme == "ssh" and port == "22")
            host = host_name if default_port else f"{host_name}:{port}"
        else:
            # Support scp-style Git remotes such as git@host:org/repo.
            host = host_name
            path = f"{port}/{path}"
    if path.endswith(".git"):
        path = path[:-4]
    if host.casefold() == "github.com":
        path = path.casefold()
    return f"{host.casefold()}/{path}"


def project_key(identity: str) -> str:
    normalized = normalize_project_identity(identity)
    repository_path = normalized.split("/", 1)[1]
    slug = re.sub(r"[^a-z0-9]+", "-", repository_path.casefold()).strip("-")
    return f"{slug}-{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:8]}"


def _legacy_runtime_normalize_project_identity(identity: str) -> str:
    """Reproduce the pre-port-fix key input for existing source directories."""

    value = identity.strip()
    value = re.sub(r"^https?://", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^git@", "", value, flags=re.IGNORECASE)
    value = value.replace(":", "/").strip("/")
    if value.endswith(".git"):
        value = value[:-4]
    host, separator, path = value.partition("/")
    if not separator or not path:
        raise FragmentValidationError("project identity must contain host and repository")
    if host.casefold() == "github.com":
        path = path.casefold()
    return f"{host.casefold()}/{path}"


def _legacy_project_key(identity: str) -> str:
    """Return the historical basename-plus-hash key used by older Fragments."""

    normalized = normalize_project_identity(identity)
    repository_name = normalized.rsplit("/", 1)[-1]
    slug = re.sub(r"[^a-z0-9]+", "-", repository_name.casefold()).strip("-")
    return f"{slug}-{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:8]}"


def _legacy_runtime_project_key(identity: str) -> str:
    normalized = _legacy_runtime_normalize_project_identity(identity)
    repository_path = normalized.split("/", 1)[1]
    slug = re.sub(r"[^a-z0-9]+", "-", repository_path.casefold()).strip("-")
    return f"{slug}-{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:8]}"


def project_key_aliases(identity: str) -> Tuple[str, ...]:
    """Return deterministic historical keys accepted during source validation."""

    canonical = project_key(identity)
    aliases = {canonical, _legacy_project_key(identity), _legacy_runtime_project_key(identity)}
    aliases.discard(canonical)
    return tuple(sorted(aliases))


def _project_directory_key(path: Path, source_root: Path) -> Optional[str]:
    relative_parts = path.resolve().relative_to(source_root.resolve()).parts
    if len(relative_parts) < 4 or relative_parts[:2] != (".memory", "projects"):
        return None
    return relative_parts[2] or None


def canonical_project_key_for_path(path: Path, source_root: Path, identity: str) -> str:
    """Resolve identity and accept only canonical or known historical keys."""

    try:
        derived_key = project_key(identity)
    except FragmentValidationError as exc:
        raise FragmentValidationError(
            str(exc), reason_code="invalid_project_identity"
        ) from exc
    path_key = _project_directory_key(path, source_root)
    accepted_keys = {derived_key, *project_key_aliases(identity)}
    if path_key not in accepted_keys:
        raise FragmentValidationError(
            f"project path key {path_key!r} does not match identity keys {sorted(accepted_keys)!r}",
            reason_code="project_key_mismatch",
        )
    return derived_key


def parse_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
    if not text.startswith("---\n"):
        raise FragmentValidationError("missing YAML front matter")
    marker = text.find("\n---\n", 4)
    if marker < 0:
        raise FragmentValidationError("unterminated YAML front matter")
    raw = text[4:marker]
    try:
        metadata = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise FragmentValidationError(f"invalid YAML front matter: {exc}") from exc
    if not isinstance(metadata, dict):
        raise FragmentValidationError("front matter must be a mapping")
    return metadata, text[marker + len("\n---\n") :]


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FragmentValidationError(f"{key} must be a non-empty string")
    return value.strip()


def _section(body: str, title: str) -> str:
    matches = list(HEADING_RE.finditer(body))
    target = None
    for match in matches:
        if match.group(2).strip().casefold() == title.casefold():
            target = match
            break
    if target is None:
        return ""
    end = len(body)
    for match in matches:
        if match.start() <= target.start():
            continue
        if len(match.group(1)) <= len(target.group(1)):
            end = match.start()
            break
    return body[target.end() : end].strip()


def _nonempty_evidence(body: str) -> bool:
    result = _section(body, "Result") or _section(body, "Task result")
    validation = _section(body, "Validation") or _section(body, "Validation summary")
    terminal_evidence = _section(body, "Evidence")
    return all(value and value.casefold() not in {"none", "n/a"} for value in (result, validation, terminal_evidence))


def _anchor_pairs(metadata: Mapping[str, Any], applicability: Applicability) -> Tuple[Tuple[str, str], ...]:
    values = []
    anchors = metadata.get("anchors") or {}
    if isinstance(anchors, Mapping):
        for anchor_type, raw_values in anchors.items():
            if isinstance(raw_values, str):
                raw_values = [raw_values]
            if isinstance(raw_values, Iterable):
                for raw_value in raw_values:
                    value = str(raw_value).strip()
                    if value:
                        values.append((str(anchor_type), value))
    for value in applicability.project_keys:
        values.append(("applies.project_key", value))
    for value in applicability.task_kinds:
        values.append(("applies.task_kind", value))
    for value in applicability.required_anchors:
        values.append(("applies.required_anchor", value))
    for value in applicability.excluded_project_keys:
        values.append(("does_not_apply.project_key", value))
    for value in applicability.excluded_task_kinds:
        values.append(("does_not_apply.task_kind", value))
    for value in applicability.excluded_anchors:
        values.append(("does_not_apply.anchor", value))
    return tuple(sorted(set(values)))


def _supersedes(metadata: Mapping[str, Any]) -> Tuple[str, ...]:
    related = metadata.get("related") or {}
    values = related.get("supersedes") if isinstance(related, Mapping) else ()
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Iterable):
        raise FragmentValidationError("related.supersedes must be a list")
    return tuple(str(value).strip() for value in values if str(value).strip())


def parse_experience_fragment(
    path: Path,
    source_root: Path,
    source_hash: str,
    created_at_fallback: str = "",
) -> ExperienceRecord:
    relative_path = path.relative_to(source_root).as_posix()
    metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
    fragment_id = _required_string(metadata, "id")
    if not ULID_RE.match(fragment_id):
        raise FragmentValidationError("id must be a 26-character Crockford ULID")
    fragment_type = _required_string(metadata, "type")
    if fragment_type not in ALLOWED_FRAGMENT_TYPES:
        raise FragmentValidationError("Experience source has an unsupported Fragment type")
    task_id_value = metadata.get("task_id")
    task_id = str(task_id_value).strip() if task_id_value else ""
    if fragment_type in FORMAL_PRIMARY_TYPES and not task_id:
        raise FragmentValidationError("new formal Task, Case, or Handoff Primary requires task_id")
    scope = _required_string(metadata, "scope")
    if scope not in {"project", "global"}:
        raise FragmentValidationError("scope must be project or global")
    identity = metadata.get("project")
    if scope == "project":
        if not isinstance(identity, str) or not identity.strip():
            raise FragmentValidationError(
                "project scope requires project identity",
                reason_code="invalid_project_identity",
            )
        stable_project_key = canonical_project_key_for_path(path, source_root, identity)
    else:
        if identity not in (None, ""):
            raise FragmentValidationError("global scope must omit project identity")
        stable_project_key = None
    status = _required_string(metadata, "status")
    title = _required_string(metadata, "title")
    summary = _required_string(metadata, "summary")
    confidence = _required_string(metadata, "confidence")
    try:
        importance = int(metadata.get("importance"))
    except (TypeError, ValueError) as exc:
        raise FragmentValidationError("importance must be an integer") from exc
    if not 1 <= importance <= 5:
        raise FragmentValidationError("importance must be between 1 and 5")
    experience = metadata.get("experience")
    if not isinstance(experience, Mapping):
        raise FragmentValidationError("missing Experience overlay")
    if not re.search(r"^##[ \t]+Experience[ \t]*$", body, re.MULTILINE) or not _section(body, "Experience"):
        raise FragmentValidationError(
            "missing Experience section",
            reason_code="missing_experience_section",
        )
    outcome = _required_string(experience, "outcome")
    if outcome not in {"NEW", "REINFORCE", "CORRECT"}:
        raise FragmentValidationError("invalid Experience outcome")
    verification = _required_string(experience, "verification")
    if verification not in {"candidate", "verified", "superseded", "invalid"}:
        raise FragmentValidationError("invalid Experience verification")
    canonical_id = _required_string(experience, "canonical_id")
    if outcome in {"NEW", "CORRECT"} and canonical_id != fragment_id:
        raise FragmentValidationError("NEW/CORRECT canonical_id must equal Fragment id")
    if outcome == "REINFORCE" and canonical_id == fragment_id:
        raise FragmentValidationError("REINFORCE canonical_id must reference an existing canonical Experience")
    capsule = _required_string(experience, "capsule")
    applies = experience.get("applies_when") or {}
    does_not_apply = experience.get("does_not_apply_when") or {}
    applicability = Applicability.from_mapping(
        {"applies_when": applies, "does_not_apply_when": does_not_apply}
    )
    feedback = None
    feedback_mapping = experience.get("feedback")
    if feedback_mapping is not None:
        if outcome != "REINFORCE" or not isinstance(feedback_mapping, Mapping) or not task_id:
            raise FragmentValidationError("feedback is only valid for REINFORCE")
        reused = feedback_mapping.get("reused")
        result = feedback_mapping.get("result")
        if reused is not True or result not in {"success", "failure"}:
            raise FragmentValidationError("REINFORCE feedback must be reused=true and result success|failure")
        if not _nonempty_evidence(body):
            raise FragmentValidationError("REINFORCE feedback lacks Task Result, Terminal Evidence, or Validation")
        feedback = FeedbackEvent(
            experience_id=fragment_id,
            canonical_id=canonical_id,
            reused=True,
            result=str(result),
            used_at=str(metadata.get("updated_at") or metadata.get("created_at") or created_at_fallback),
        )
    if outcome == "REINFORCE" and feedback is None:
        raise FragmentValidationError("REINFORCE requires rebuildable feedback")
    created_at = str(metadata.get("created_at") or created_at_fallback)
    if not created_at:
        raise FragmentValidationError("created_at is required")
    return ExperienceRecord(
        id=fragment_id,
        task_id=task_id,
        scope=scope,
        project_key=stable_project_key,
        kind=str(experience.get("kind") or fragment_type),
        fragment_status=status,
        experience_verification=verification,
        title=title,
        summary=summary,
        capsule=capsule,
        confidence=confidence,
        importance=importance,
        source_path=relative_path,
        source_hash=source_hash,
        created_at=created_at,
        updated_at=str(metadata.get("updated_at")) if metadata.get("updated_at") else None,
        outcome=outcome,
        canonical_id=canonical_id,
        applicability=applicability,
        trigger=_section(body, "Trigger"),
        core_action=" ".join(value for value in (_section(body, "Worked"), _section(body, "Reuse")) if value),
        does_not_apply=_section(body, "Does not apply when"),
        exceptions=_section(body, "Exceptions"),
        anchors=_anchor_pairs(metadata, applicability),
        supersedes=_supersedes(metadata),
        feedback=feedback,
    )
