#!/usr/bin/env python3
"""汇总结构/覆盖/忠实三项检查，给出评分与判定。"""
from __future__ import annotations
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def main() -> int:
    data = json.loads(os.environ["DAG8_INPUTS"])
    session = json.loads(data["session"])
    st = json.loads(data["structure"]); cov = json.loads(data["coverage"])
    fa = json.loads(data["faithfulness"])
    chars = st["chars"]
    readability = 1.0 if 800 <= chars <= 8000 else (0.7 if 300 <= chars <= 15000 else 0.4)
    if st["code_ratio"] > 0.45: readability -= 0.2
    if not st["has_summary"]: readability -= 0.1
    readability = max(0.0, min(1.0, readability))
    score = (0.35 * cov["ratio"] + 0.25 * st["score"]
             + 0.25 * fa["score"] + 0.15 * readability)
    verdict = "good" if score >= 0.75 else ("needs_work" if score >= 0.55 else "bad")
    findings = []
    if st["empty_sections"]:
        findings.append(f"空/占位章节：{', '.join(st['empty_sections'])}")
    if st["placeholder_count"]:
        findings.append(f"占位符出现 {st['placeholder_count']} 次")
    if st["nested_h2_from_original"]:
        findings.append("原文标题未降级，和报告章节同级")
    if not st["has_summary"]:
        findings.append("缺少结论先行/摘要")
    missing = [d["label"] for d in cov["details"] if not d["ok"]]
    if missing:
        findings.append("覆盖不足：" + "、".join(missing))
    if fa["supported_ratio"] < 0.5:
        findings.append(f"与 session 原文重合度低：{fa['supported_ratio']:.2f}")
    if not fa["session_id_present"]:
        findings.append("文档未带 session id")
    if not fa["source_present"]:
        findings.append("文档未带可追溯 source")

    lines = [
        "# 结论文档质量评判", "",
        f"- 会话：`{session.get('session_id', '')}`",
        f"- 文档：`{st['path']}`",
        f"- 字数：{chars}",
        f"- **总分：{score:.2f} / 1.00**",
        f"- **判定：{verdict}**", "",
        "## 分项", "",
        "| 维度 | 分数 | 事实 |", "|---|---:|---|",
        f"| 覆盖 | {cov['ratio']:.2f} | {cov['covered']}/{cov['total']} 个维度非空 |",
        f"| 结构 | {st['score']:.2f} | 空章节 {len(st['empty_sections'])}，占位 {st['placeholder_count']} |",
        f"| 忠实/可追溯 | {fa['score']:.2f} | 原文重合 {fa['supported_ratio']:.2f}，session id {fa['session_id_present']} |",
        f"| 可读性 | {readability:.2f} | {chars} 字，summary={st['has_summary']} |", "",
        "## 发现", "",
    ]
    lines += [f"- {f}" for f in findings] or ["- 无硬性问题"]
    lines += ["", "## 建议", ""]
    recs = []
    if missing: recs.append("补齐空维度，或明确写“本次未覆盖”并说明原因。")
    if st["nested_h2_from_original"]: recs.append("把原文标题整体降级，避免和报告章节同级。")
    if not st["has_summary"]: recs.append("在开头增加 3–5 句一句话结论。")
    if not fa["source_present"]: recs.append("增加 `dsh-session://<id>` 可追溯来源。")
    if not recs: recs.append("结构完整，可进入人工内容复核。")
    lines += [f"- {r}" for r in recs]

    slug = Path(st["path"]).stem.replace("session_", "")[:24]
    out = ROOT / "reports" / f"session_doc_eval_{slug}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps({
        "score": round(score, 3), "verdict": verdict, "findings": findings,
        "structure": st, "coverage": cov, "faithfulness": fa,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "verdict": verdict, "score": round(score, 3),
                      "report": str(out), "findings": findings}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
