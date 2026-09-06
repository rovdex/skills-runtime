# Shared Memory Runtime

This is the external Phase 2 Runtime for the Experience overlay defined by
the Shared Knowledge repository at `<CODEX_HOME>/skills`.

The Runtime is derived and local. Markdown Fragments remain authoritative;
Git history proves source revision and Remote persistence; SQLite only finds,
ranks, caches, and projects eligible Capsules.

The package provides:

- `ExperienceProjector.rebuild()` for Markdown/Git to SQLite rebuilding;
- deterministic source-hash and verified-remote proof;
- scope, applicability, FTS5/trigram/fallback Recall with bounded Capsules;
- effective verification from verified `CORRECT` supersedes edges; and
- one deterministic `compile_terminal_experience()` operation for NEW /
  REINFORCE / CORRECT classification.

Portable Knowledge reference APIs read `<CODEX_HOME>/skills/.knowledge` directly:

- `recall_skills()` implements the authoritative Skill-first, Capsule-first
  Recall path with explicit Full Skill expansion and Experience fallback;
- `validate_knowledge_tree()` checks index/Markdown/evidence parity; and
- `skill_evolution_check()` computes only the frozen evolution decisions from
  Shared Knowledge evidence.

These APIs are optional accelerators and validators. Portable Recall remains
complete with Agent-native filesystem, Git, Markdown, YAML, and JSONL tools
when this Runtime, SQLite, or a Python helper is absent. Candidate Skill
guidance never changes J3; only Verified Skill advice is eligible for the
allowlisted Knowledge Adaptation path.

It also provides a read-only Finalization Receipt check for the Final Summary
gate. From the Runtime checkout, run:

```text
PYTHONPATH=src python -m shared_memory_runtime \
  --source-root <CODEX_HOME>/skills receipt --state <owning-state-path>
```

The command returns JSON and exits zero only when the State has a legal terminal
pair, exactly one valid task-owned Primary is present, and that Primary is
proved on the configured Shared Knowledge Remote.

Cross-device reconciliation is separate from the terminal Receipt gate. Run:

```text
PYTHONPATH=src python -m shared_memory_runtime \
  --source-root <CODEX_HOME>/skills reconcile \
  --state-root <CODEX_HOME>/.state/finalization \
  --task-id <task-id>
```

The read-only result may report `Stale Local Finalization State` with
`shared_evidence_complete: true` and `local_terminal_receipt: not_proven`.
This never claims a remote Receipt was executed and never writes local State.
Only the existing `receipt` command can report `complete: true`.

No daemon, hook, background service, vector store, embedding dependency, or
additional Runtime State system is introduced.

## Development

```text
PYTHONPATH=src python3 tests/run_tests.py
PYTHONPATH=src python3 benchmarks/phase3_baseline.py
```

When pytest is available, `python3 -m pytest` runs the same test module.

The normal database location is `<CODEX_HOME>/.state/experience.db`; tests use
temporary databases and repositories.

Phase 3 baseline measurements are emitted as separate local reports for Local
Runtime Performance and Remote Network Performance. Formal Task metrics use a
single best-effort append to `<CODEX_HOME>/.state/experience-runtime/metrics.jsonl`.

## Cross-device Bootstrap Compatibility

The existing `skill-sync` helper provisions this repository at
`<CODEX_HOME>/.runtime`; the Shared Knowledge checkout remains
`<CODEX_HOME>/skills`. Runtime Compatibility validates this Git worktree,
entrypoints, imports, Python/SQLite/FTS5 support, and the current schema. It
does not require `<CODEX_HOME>/.state/experience.db` to exist.

Projection Compatibility is checked separately through
`ExperienceProjector.projection_freshness(shared_knowledge_fresh=True)` after
formal Shared Knowledge Freshness. Without that in-memory prerequisite the
checkout is not authoritative. A missing database is
`projection_missing`, not Runtime incompatibility. A fresh source set and
source hashes reuse the projection; a missing, corrupt, or stale projection
uses the controlled authoritative Markdown/Git rebuild with the existing
quarantine and atomic install recovery. Bootstrap readiness caches only a
complete successful result in process memory; failures remain retryable.
Rebuild and Freshness use the same canonical identity/path resolver. A
malformed or path-mismatched project source is retained as a non-blocking
diagnostic and excluded from the eligible source set.

The projection keeps the frozen `experiences` columns explicit. `canonical_id`
and `outcome` are retained as reserved identity anchor rows in
`experience_anchors`, so canonical deduplication and learning outcomes remain
rebuildable without adding ambiguous status columns or a new Memory type.
