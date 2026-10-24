#!/usr/bin/env python3
"""检查 README 里的测试计数与 pytest 实际收集是否一致（防文档漂移）。

    python scripts/check_docs.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout

    actual: Counter = Counter()
    for line in out.splitlines():
        if "::" not in line:
            continue
        name = line.split("::")[0].split("/")[-1]
        if name.startswith("test_") and name.endswith(".py"):
            actual[name] += 1

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    documented = {}
    for line in readme.splitlines():
        if line.startswith("| `tests/test_") and ".py`" in line:
            parts = [p.strip() for p in line.split("|")]
            documented[parts[1].strip("`").split("/")[-1]] = int(parts[2])

    problems = []
    for name, n in sorted(actual.items()):
        if documented.get(name) != n:
            problems.append(f"{name}: 实际 {n}，README {documented.get(name)}")
    for name in sorted(set(documented) - set(actual)):
        problems.append(f"{name}: README 有、实际没有")

    total = sum(actual.values())
    m = re.search(r"全量 (\d+) 条", readme)
    if m and int(m.group(1)) != total:
        problems.append(f"总数：实际 {total}，README {m.group(1)}")

    if problems:
        print("文档计数不一致：")
        for p in problems:
            print("  -", p)
        return 1

    print(f"文档计数一致：{len(actual)} 个测试文件 / {total} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
