#!/usr/bin/env python3
"""Mechanical checks for docs/*.md, implementing docs/DOC_STANDARDS.md.

Stdlib only, so it can run in CI before any dependency is installed. Rules that
cannot be checked mechanically (e.g. "every command was actually run") live in
the standard document itself.

Usage:  python3 scripts/check_docs.py [root]
Exit:   0 when no rule fails; 1 otherwise. Warnings never fail the run.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Rules that only apply to *current* docs. Archives are records, not references,
# so they are exempt from everything except fence balance.
ARCHIVE_DIR = "archive"
INDEX = "docs/README.md"
CURRENT = (INDEX, "README.md")
EXEMPT_STALE = "CODE_REVIEW"  # quotes stale strings on purpose, as findings

FENCE = re.compile(r"^\s*(```|~~~)")
LINK = re.compile(r"\]\(([^)]+)\)")
ABS_PATH = re.compile(r"(/home/[A-Za-z0-9_.-]+/|/mnt/[A-Za-z0-9_.-]+/|file:///|[A-Z]:\\)")
STALE_COUNT = re.compile(r"(\d{2,})\s+(?:automated\s+)?tests?\s+(?:passing|pass\b|pass rate)")


def strip_spans(line: str) -> str:
    """Drop inline code and quoted spans, so *quoted* evidence is not flagged."""
    out = line
    for pattern in (r"`[^`]*`", r'"[^"]*"'):
        out = re.sub(pattern, "", out)
    return out


def markdown_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"))


def check_fences(root: Path) -> list[tuple[str, str]]:
    """Every fenced block must be closed."""
    problems = []
    for path in markdown_files(root):
        stack = []
        for line in path.read_text(encoding="utf-8").splitlines():
            m = FENCE.match(line)
            if not m:
                continue
            marker = m.group(1)
            if stack and stack[-1] == marker:
                stack.pop()
            else:
                stack.append(marker)
        if stack:
            problems.append((str(path.relative_to(root)), f"{len(stack)} unclosed fence(s)"))
    return problems


def check_language_tags(root: Path) -> list[tuple[str, str]]:
    """Warn on untagged fences (warnings never fail)."""
    warnings = []
    for path in markdown_files(root):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            m = FENCE.match(line)
            if m and len(line.strip()) == 3:  # "```" with nothing after it
                warnings.append((str(path.relative_to(root)), f"line {i}: fence has no language tag"))
    return warnings


def check_links(root: Path) -> list[tuple[str, str]]:
    """Every relative markdown link in current docs must resolve.

    Archives are exempt: their links point at the layout that existed when they
    were written, and rewriting them would falsify the record.
    """
    broken = []
    for path in markdown_files(root):
        if ARCHIVE_DIR in path.parts:
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            for target in LINK.findall(strip_spans(raw)):
                target = target.split("#")[0].strip()
                if not target or re.match(r"[a-z]+://", target) or target.startswith("mailto:"):
                    continue
                candidate = (path.parent / target).resolve()
                try:
                    candidate.relative_to(root.resolve())
                except ValueError:
                    broken.append((str(path.relative_to(root)), f"escapes the repo: {target}"))
                    continue
                if not candidate.exists():
                    broken.append((str(path.relative_to(root)), f"missing target: {target}"))
    return broken


def check_absolute_paths(root: Path) -> list[tuple[str, str]]:
    """No local-machine paths in current docs (quoted findings are fine)."""
    problems = []
    for path in markdown_files(root):
        if ARCHIVE_DIR in path.parts or EXEMPT_STALE in path.name:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            cleaned = strip_spans(line)
            match = ABS_PATH.search(cleaned)
            if match:
                problems.append(
                    (str(path.relative_to(root)), f"line {i}: local absolute path {match.group(1)!r}")
                )
    return problems


def check_stale_counts(root: Path) -> list[tuple[str, str]]:
    """Never claim "N tests passing" — state the command instead."""
    problems = []
    for path in markdown_files(root):
        if ARCHIVE_DIR in path.parts or EXEMPT_STALE in path.name:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = STALE_COUNT.search(strip_spans(line))
            if match:
                problems.append((str(path.relative_to(root)), f"line {i}: stale count phrasing {match.group(0)!r}"))
    return problems


def check_index_coverage(root: Path) -> list[tuple[str, str]]:
    """Every non-archived docs/*.md must be listed in docs/README.md."""
    index = root / INDEX
    if not index.exists():
        return [(INDEX, "index file is missing")]
    listed = index.read_text(encoding="utf-8")
    missing = []
    for path in sorted((root / "docs").glob("*.md")):
        if path.name == INDEX or path.name.endswith(".py"):
            continue
        if path.name not in listed:
            missing.append(path.name)
    if not missing:
        return []
    return [(INDEX, "not listed: " + ", ".join(missing))]


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else Path(__file__).resolve().parent.parent).resolve()
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 1

    rules = [
        ("fences", check_fences(root), "fail"),
        ("links", check_links(root), "fail"),
        ("absolute paths", check_absolute_paths(root), "fail"),
        ("stale counts", check_stale_counts(root), "fail"),
        ("index coverage", check_index_coverage(root), "fail"),
        ("language tags", check_language_tags(root), "warn"),
    ]

    failures = 0
    warnings = 0
    for name, problems, severity in rules:
        if not problems:
            print(f"  ok  {name}")
            continue
        if severity == "fail":
            failures += len(problems)
        else:
            warnings += len(problems)
        verb = "FAIL" if severity == "fail" else "warn"
        print(f"  {verb}  {name} ({len(problems)})")
        for where, detail in problems[:12]:
            print(f"        {where}: {detail}")
        if len(problems) > 12:
            print(f"        ... and {len(problems) - 12} more")

    print(f"\n{root}: {failures} failure(s), {warnings} warning(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
