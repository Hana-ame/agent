#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""save_md.py — DAG 的落盘节点：从 stdin 读定稿 Markdown，写到 out/ 下。

只做三件事：
1. 兜底校验：必须像一篇完整的深度笔记（有一级标题与四个一级章节）；
2. 写入 out/troll_tea_party_notes.md（固定名，便于对比/引用）；
3. 同时写 out/latest.md（最新一次，方便脚本取）。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def main() -> int:
    text = sys.stdin.read().strip()
    if not text:
        print("save_md: stdin 为空", file=sys.stderr)
        return 1
    if not text.startswith("# "):
        print("save_md: 定稿没有一级标题", file=sys.stderr)
        return 1
    required = [
        "## 一、",
        "## 二、",
        "## 三、",
        "## 四、",
    ]
    missing = [h for h in required if h not in text]
    if missing:
        print(f"save_md: 缺少章节 {missing}", file=sys.stderr)
        return 1
    if "```mermaid" not in text:
        print("save_md: 缺少 mermaid 代码块", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "troll_tea_party_notes.md"
    target.write_text(text + "\n", encoding="utf-8")
    (OUT / "latest.md").write_text(text + "\n", encoding="utf-8")
    print(str(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
