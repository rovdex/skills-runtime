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
`ExperienceProjector.projection_freshness()`. A missing database is
`projection_missing`, not Runtime incompatibility. A fresh source set and
source hashes reuse the projection; a missing, corrupt, or stale projection
uses the controlled authoritative Markdown/Git rebuild with the existing
quarantine and atomic install recovery. Bootstrap readiness caches only a
complete successful result in process memory; failures remain retryable.

The projection keeps the frozen `experiences` columns explicit. `canonical_id`
and `outcome` are retained as reserved identity anchor rows in
`experience_anchors`, so canonical deduplication and learning outcomes remain
rebuildable without adding ambiguous status columns or a new Memory type.
