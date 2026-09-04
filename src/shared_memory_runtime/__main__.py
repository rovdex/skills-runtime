"""Small explicit CLI for rebuild and bounded Recall."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .projector import ExperienceProjector
from .recall import RecallContext
from .finalization_receipt import verify_finalization_receipt
from .finalization_reconciliation import reconcile_finalization_states


def main() -> int:
    parser = argparse.ArgumentParser(description="Shared Memory Experience Runtime")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--database", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("rebuild")
    recall_parser = subparsers.add_parser("recall")
    recall_parser.add_argument("--project-key")
    recall_parser.add_argument("--task-kind")
    recall_parser.add_argument("--query", default="")
    recall_parser.add_argument("--anchor", action="append", default=[])
    recall_parser.add_argument("--task-class", choices=["normal", "complex", "debugging"], default="normal")
    recall_parser.add_argument("--expand", action="store_true")
    receipt_parser = subparsers.add_parser("receipt", help="verify formal Task terminal evidence without writes")
    receipt_parser.add_argument("--state", type=Path, required=True)
    reconcile_parser = subparsers.add_parser(
        "reconcile", help="reconcile local Finalization State with shared evidence without writes"
    )
    reconcile_parser.add_argument("--state-root", type=Path, required=True)
    reconcile_parser.add_argument("--state", type=Path, action="append", default=[])
    reconcile_parser.add_argument("--task-id", action="append", default=[])
    arguments = parser.parse_args()
    projector = ExperienceProjector(arguments.source_root, arguments.database)
    if arguments.command == "rebuild":
        report = projector.rebuild()
        print(
            json.dumps(
                {
                    "projected": len(report.projected),
                    "remote_verified": report.remote_verified_count,
                    "skipped": len(report.skipped),
                    "skip_reason_counts": Counter(item.reason_code for item in report.skipped),
                    "skipped_sources": [item.as_mapping() for item in report.skipped],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if arguments.command == "receipt":
        receipt = verify_finalization_receipt(arguments.source_root, arguments.state)
        print(json.dumps(receipt.as_mapping(), ensure_ascii=False))
        return 0 if receipt.complete else 1
    if arguments.command == "reconcile":
        report = reconcile_finalization_states(
            arguments.source_root,
            arguments.state_root,
            state_paths=tuple(arguments.state),
            task_ids=tuple(arguments.task_id),
        )
        print(json.dumps(report.as_mapping(), ensure_ascii=False))
        return 0
    candidates = projector.recall(
        RecallContext(
            project_key=arguments.project_key,
            task_kind=arguments.task_kind,
            query=arguments.query,
            anchors=tuple(arguments.anchor),
            task_class=arguments.task_class,
            expand=arguments.expand,
        )
    )
    print(json.dumps([candidate.__dict__ for candidate in candidates], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
