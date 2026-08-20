#!/usr/bin/env python3
"""Validate the course-only branch without depending on the engine checkout."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "apps",
    "benchmarks",
    "bindings",
    "include",
    "examples",
    "python",
    "scripts",
    "src",
    "tests",
    "CMakeLists.txt",
}
REQUIRED = {
    "README.md",
    "docs/DESIGN_FOR_BEGINNERS.zh-CN.md",
    "docs/OPERATOR_CONTRACTS.zh-CN.md",
    "docs/TASK_CONTRACT.md",
    "notebooks/README.md",
    *(f"notebooks/N{index}_{name}.md" for index, name in enumerate((
        "storage_tensor",
        "cpu_hip",
        "autograd",
        "checkpoint",
        "transformer",
        "training_generation",
        "performance",
        "multi_gpu",
        "evidence_atlas",
    ))),
    "pa/PA0/README.md",
    "pa/PA1/README.md",
    "pa/PA2/README.md",
}
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return {line for line in result.stdout.splitlines() if line}


def check_layout(tracked: set[str], errors: list[str]) -> None:
    for relative in sorted(FORBIDDEN):
        if relative in tracked or any(path.startswith(f"{relative}/") for path in tracked):
            errors.append(f"framework artifact is forbidden on course branch: {relative}")
    for relative in sorted(REQUIRED):
        if not (ROOT / relative).is_file():
            errors.append(f"required course file is missing: {relative}")


def check_links(tracked: set[str], errors: list[str]) -> int:
    checked = 0
    for relative_document in sorted(path for path in tracked if path.endswith(".md")):
        document = ROOT / relative_document
        text = document.read_text(encoding="utf-8")
        for raw_target in LINK.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path_text = unquote(target.split("#", 1)[0])
            candidate = (document.parent / path_text).resolve()
            checked += 1
            if not candidate.exists():
                relative = document.relative_to(ROOT)
                errors.append(f"broken local link in {relative}: {target}")
    return checked


def main() -> int:
    errors: list[str] = []
    tracked = tracked_files()
    check_layout(tracked, errors)
    checked_links = check_links(tracked, errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    markdown_count = sum(path.endswith(".md") for path in tracked)
    print(
        f"course validation passed: {len(REQUIRED)} required files, "
        f"{markdown_count} Markdown files, {checked_links} local links"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
