#!/usr/bin/env python3
"""覆盖检查：6 个复盘维度是否都有非空章节。"""
from __future__ import annotations
import json, os, re

EXPECT = [
    ("background", "背景与用户诉求", ("背景", "诉求", "用户")),
    ("process", "过程与关键动作", ("过程", "动作", "执行", "工具")),
    ("decisions", "技术决策与判断", ("决策", "判断", "取舍")),
    ("conclusion", "最终结论与证据", ("结论", "证据", "事实")),
    ("risks", "风险与未决", ("风险", "未决", "遗留")),
    ("reuse", "可复用经验与下一步", ("复用", "经验", "下一步", "建议")),
]
PLACEHOLDER = ("未生成", "未覆盖", "占位", "TODO", "TBD", "（无）", "(无)")

def main() -> int:
    data = json.loads(os.environ["DAG8_INPUTS"])
    session = json.loads(data["session"]); doc = json.loads(data["doc"])["text"]
    sections = []
    cur = None
    for ln in doc.splitlines():
        if ln.startswith("## "):
            if cur: sections.append(cur)
            cur = {"title": ln[3:].strip(), "body": []}
        elif cur is not None:
            cur["body"].append(ln)
    if cur: sections.append(cur)
    details, covered = [], 0
    for key, label, words in EXPECT:
        hit = None
        for s in sections:
            if any(w in s["title"] for w in words):
                hit = s; break
        body = "\n".join(hit["body"]).strip() if hit else ""
        ok = bool(hit) and len(body) >= 20 and not any(p in body for p in PLACEHOLDER)
        if ok: covered += 1
        details.append({"key": key, "label": label, "found": bool(hit),
                        "chars": len(body), "ok": ok})
    ratio = covered / len(EXPECT)
    print(json.dumps({"covered": covered, "total": len(EXPECT), "ratio": round(ratio, 3),
                      "details": details}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
