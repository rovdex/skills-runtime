"""Authoritative Git source and remote-persistence evidence."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class RemoteProof:
    verified: bool
    source_path: str
    source_hash: Optional[str]
    containing_revision: Optional[str]
    remote_name: Optional[str]
    remote_head: Optional[str]
    reason_code: str


class GitCommandError(RuntimeError):
    """A Git command failed while collecting evidence."""


class GitEvidence:
    """Read-only Git evidence collector for a Shared Knowledge source tree."""

    def __init__(self, repository_root: Path, remote_snapshot: Optional[tuple[str, str]] = None):
        self.repository_root = Path(repository_root).resolve()
        self.remote_snapshot = remote_snapshot

    def _run(self, args: Sequence[str], check: bool = True) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.repository_root), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if check and result.returncode != 0:
            raise GitCommandError(
                f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
            )
        return result.stdout.strip()

    def is_repository(self) -> bool:
        result = subprocess.run(
            ["git", "-C", str(self.repository_root), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.returncode == 0

    def current_branch(self) -> Optional[str]:
        branch = self._run(["branch", "--show-current"], check=False)
        return branch or None

    def upstream(self) -> Optional[str]:
        value = self._run(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
            check=False,
        )
        return value or None

    def tracked_source_hash(self, source_path: str) -> Optional[str]:
        try:
            self._run(["ls-files", "--error-unmatch", "--", source_path])
            value = self._run(["hash-object", "--", source_path])
        except GitCommandError:
            return None
        return value or None

    def blob_at(self, revision: str, source_path: str) -> Optional[str]:
        value = self._run(["rev-parse", f"{revision}:{source_path}"], check=False)
        return value or None

    def remote_head(self) -> tuple[Optional[str], Optional[str], Optional[str]]:
        if self.remote_snapshot is not None:
            remote_name, head = self.remote_snapshot
            return remote_name, head, "snapshot"
        upstream = self.upstream()
        if not upstream or "/" not in upstream:
            return None, None, "upstream_unavailable"
        remote_name, branch = upstream.split("/", 1)
        try:
            remote_url = self._run(["remote", "get-url", remote_name])
            if not remote_url:
                return None, None, "remote_url_unavailable"
            result = subprocess.run(
                ["git", "-C", str(self.repository_root), "ls-remote", remote_name, f"refs/heads/{branch}"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except GitCommandError:
            return None, None, "remote_url_unavailable"
        if result.returncode != 0:
            return remote_name, None, "remote_head_unavailable"
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == f"refs/heads/{branch}":
                return remote_name, parts[0], "ok"
        return remote_name, None, "remote_head_unavailable"

    def containing_revision(self, remote_head: str, source_path: str, source_hash: str) -> Optional[str]:
        revisions = self._run(["rev-list", remote_head, "--", source_path], check=False)
        for revision in revisions.splitlines():
            if self.blob_at(revision, source_path) == source_hash:
                return revision
        return None

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        result = subprocess.run(
            ["git", "-C", str(self.repository_root), "merge-base", "--is-ancestor", ancestor, descendant],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.returncode == 0

    def prove_remote_persistence(
        self,
        source_path: str,
        expected_source_hash: Optional[str] = None,
    ) -> RemoteProof:
        source_hash = self.tracked_source_hash(source_path)
        if not source_hash:
            return RemoteProof(False, source_path, None, None, None, None, "source_not_tracked")
        if expected_source_hash and source_hash != expected_source_hash:
            return RemoteProof(False, source_path, source_hash, None, None, None, "source_hash_mismatch")
        remote_name, verified_remote_head, remote_reason = self.remote_head()
        if not verified_remote_head:
            return RemoteProof(
                False,
                source_path,
                source_hash,
                None,
                remote_name,
                None,
                remote_reason,
            )
        try:
            self._run(["cat-file", "-e", f"{verified_remote_head}^{{commit}}"])
        except GitCommandError:
            return RemoteProof(
                False,
                source_path,
                source_hash,
                None,
                remote_name,
                verified_remote_head,
                "remote_head_object_unavailable",
            )
        containing = self.containing_revision(verified_remote_head, source_path, source_hash)
        if not containing:
            return RemoteProof(
                False,
                source_path,
                source_hash,
                None,
                remote_name,
                verified_remote_head,
                "containing_revision_unavailable",
            )
        if not self.is_ancestor(containing, verified_remote_head):
            return RemoteProof(
                False,
                source_path,
                source_hash,
                containing,
                remote_name,
                verified_remote_head,
                "containing_revision_not_reachable",
            )
        return RemoteProof(
            True,
            source_path,
            source_hash,
            containing,
            remote_name,
            verified_remote_head,
            "verified",
        )
