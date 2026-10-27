#!/usr/bin/env python3
"""结构检查：标题层级、空章节、占位符、总结、代码占比。"""
from __future__ import annotations
import json, os, re

PLACEHOLDER = ("未生成", "未覆盖", "占位", "TODO", "TBD", "（无）", "(无)")

def main() -> int:
    data = json.loads(os.environ["DAG8_INPUTS"])
    doc = json.loads(data["doc"])
    text = doc["text"]
    lines = text.splitlines()
    heads = [(i, l) for i, l in enumerate(lines) if re.match(r"^#{1,6}\s+", l)]
    h1 = [l for _, l in heads if l.startswith("# ")]
    h2 = [l for _, l in heads if l.startswith("## ")]
    # 按 h2 切段
    sections = []
    cur = None
    for ln in lines:
        if ln.startswith("## "):
            if cur: sections.append(cur)
            cur = {"title": ln[3:].strip(), "lines": []}
        elif cur is not None:
            cur["lines"].append(ln)
    if cur: sections.append(cur)
    empty = []
    for s in sections:
        body = "\n".join(s["lines"]).strip()
        if len(body) < 20 or any(p in body for p in PLACEHOLDER):
            empty.append(s["title"])
    placeholder_count = sum(text.count(p) for p in PLACEHOLDER)
    code_chars = sum(len(m.group(0)) for m in re.finditer(r"```.*?```", text, re.S))
    code_ratio = round(code_chars / max(1, len(text)), 3)
    has_summary = any(("总结" in s["title"] or "摘要" in s["title"] or "结论先行" in s["title"])
                      and len("\n".join(s["lines"]).strip()) >= 20 for s in sections)
    # 结论章节里出现多个同级 h2，说明原始标题没降级
    nested_h2 = len([x for x in h2 if x.startswith("## 目录") or x.startswith("## 三层")
                     or x.startswith("## 运行链路") or x.startswith("## 加新")])
    score = 1.0
    score -= min(0.5, 0.12 * len(empty))
    score -= min(0.2, 0.03 * placeholder_count)
    score += 0.1 if has_summary else -0.1
    score -= 0.15 if nested_h2 else 0
    score -= 0.1 if code_ratio > 0.45 else 0
    print(json.dumps({
        "path": doc["path"], "chars": len(text), "lines": len(lines),
        "h1": h1, "h2": h2, "section_count": len(sections),
        "empty_sections": empty, "placeholder_count": placeholder_count,
        "code_ratio": code_ratio, "has_summary": has_summary,
        "nested_h2_from_original": nested_h2,
        "score": round(max(0.0, min(1.0, score)), 3),
    }, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
