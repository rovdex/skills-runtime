"""Dependency-light test runner for environments without pytest."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("runtime_tests", Path(__file__).with_name("test_runtime.py"))
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> int:
    functions = [
        getattr(MODULE, name)
        for name in sorted(dir(MODULE))
        if name.startswith("test_")
    ]
    for function in functions:
        if inspect.signature(function).parameters:
            with tempfile.TemporaryDirectory(prefix="shared-memory-runtime-test-") as directory:
                function(Path(directory))
        else:
            function()
    print(f"passed={len(functions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
