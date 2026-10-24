#!/usr/bin/env python3
"""核对文档声明：账本数字、测试总数、相对链接是否存在。

    python scripts/check_claims.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", ROOT / "STATUS.md"]


def _test_total() -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    m = re.search(r"^(\d+) tests? collected", out, re.M)
    return int(m.group(1)) if m else 0


def _check_links(doc: Path, problems: list[str]) -> None:
    text = doc.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        target = target.split("#", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        if "{" in target or "*" in target:
            continue
        p = (doc.parent / target).resolve()
        if not p.exists():
            problems.append(f"{doc.name}: 链接不存在 {target}")


def main() -> int:
    ledger = json.loads((ROOT / "data" / "evolution" / "ledger.json").read_text(encoding="utf-8"))
    n, valid = ledger["n_evaluations"], ledger["n_valid"]
    total = _test_total()

    problems: list[str] = []

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"### 实测（[^）]*?(\d+) 次评估 / 有效 (\d+) / 失败 (\d+)）", readme)
    if not m:
        problems.append("README: 找不到当前账本段（### 实测）")
    elif (int(m.group(1)), int(m.group(2))) != (n, valid):
        problems.append(
            f"README: 当前账本写 {m.group(1)}/{m.group(2)}，实际 {n}/{valid}")

    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    m = re.search(r"GA 账本：\*\*(\d+) 次（(\d+) 有效", status)
    if not m:
        problems.append("STATUS: 找不到当前 GA 账本行")
    elif (int(m.group(1)), int(m.group(2))) != (n, valid):
        problems.append(
            f"STATUS: 账本写 {m.group(1)}/{m.group(2)}，实际 {n}/{valid}")

    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for m in re.finditer(r"全量 (\d+) 条", text):
            if int(m.group(1)) != total:
                problems.append(f"{doc.name}: 测试写 {m.group(1)} 条，实际 {total}")
        _check_links(doc, problems)

    if problems:
        print("文档声明不一致：")
        for p in problems:
            print("  -", p)
        return 1
    print(f"文档声明一致：账本 {n}/{valid}，测试 {total}，链接全部存在")
    return 0


if __name__ == "__main__":
    sys.exit(main())
