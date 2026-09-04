# Copyright (c) 2026 Nick van der Merwe
"""Enforce the production source and per-file line budgets."""

from __future__ import annotations

import sys
from pathlib import Path

TOTAL_LIMIT = 3_000
FILE_LIMIT = 500
SOURCE_ROOTS = (Path("src"), Path("sandbox/src"))
SCRIPT_FILES = (
    Path("sandbox/codex-as-user"),
    Path("sandbox/wiseman-discord"),
    Path("sandbox/wiseman-image"),
)


def source_files() -> list[Path]:
    """Return tracked production source paths, excluding caches and artifacts."""
    files = [path for root in SOURCE_ROOTS for path in root.rglob("*.py")]
    return sorted(files + [path for path in SCRIPT_FILES if path.exists()])


def source_lines(path: Path) -> int:
    """Count nonblank, non-comment source lines as the repository LoC measure."""
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def main() -> int:
    """Print the budget report and fail when either limit is exceeded."""
    counts = [(path, source_lines(path)) for path in source_files()]
    total = sum(lines for _, lines in counts)
    largest = sorted(counts, key=lambda item: item[1], reverse=True)
    sys.stdout.write(f"production LoC: {total}/{TOTAL_LIMIT}\n")
    sys.stdout.write(f"largest file: {largest[0][0]} ({largest[0][1]} lines)\n")
    violations = [(path, lines) for path, lines in counts if lines > FILE_LIMIT]
    if total >= TOTAL_LIMIT or violations:
        for path, lines in violations:
            sys.stderr.write(f"file budget exceeded: {path} ({lines}>{FILE_LIMIT})\n")
        if total >= TOTAL_LIMIT:
            sys.stderr.write(f"total budget exceeded: {total}>={TOTAL_LIMIT}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
