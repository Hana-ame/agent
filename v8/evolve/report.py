"""多次试验的横向报告：长度/结构/相似度/稳定性。"""
from __future__ import annotations

import difflib
import itertools
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List


def _section(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    return text[i:j] if j > 0 else text[i:]


def _sim(a: str, b: str) -> float:
    """对称相似度：SequenceMatcher 的 ratio 不是严格对称的，取双向平均。"""
    return (difflib.SequenceMatcher(None, a, b).ratio()
            + difflib.SequenceMatcher(None, b, a).ratio()) / 2


def _metrics(text: str) -> Dict[str, Any]:
    ascii_block = ""
    i = text.find("```text")
    if i >= 0:
        j = text.find("```", i + 7)
        ascii_block = text[i:j] if j > 0 else ""
    mermaid_block = ""
    i = text.find("```mermaid")
    if i >= 0:
        j = text.find("```", i + 10)
        mermaid_block = text[i:j] if j > 0 else ""
    t31 = _section(text, "### 1. 案例对标", "### 2.")
    t34 = _section(text, "### 4. 原文口语", "## 四、")
    return {
        "chars": len(text),
        "headings": len(re.findall(r"(?m)^#{1,6}\s+\S", text)),
        "stages": len(re.findall(r"(?m)^\d+\. \*\*第[一二三四五]阶段", text)),
        "t31_rows": t31.count("\n|"),
        "t34_rows": t34.count("\n|"),
        "ascii_lines": ascii_block.count("\n"),
        "mermaid_lines": mermaid_block.count("\n"),
    }


def render_trials(run_dir: str | Path) -> Path:
    base = Path(run_dir)
    review = json.loads((base / "review.json").read_text(encoding="utf-8"))
    ratings = {}
    if (base / "ratings.json").is_file():
        ratings = json.loads((base / "ratings.json").read_text(encoding="utf-8")).get("ratings", {})

    rows: List[Dict[str, Any]] = []
    for c in review["candidates"]:
        p = Path(c.get("output_path", ""))
        text = p.read_text(encoding="utf-8") if p.is_file() else ""
        m = _metrics(text)
        leak = c.get("leakage") or {}
        rows.append({
            "id": c["candidate_id"],
            "name": c.get("name", ""),
            "auto": (c.get("auto") or {}).get("score"),
            "leak": "未启用" if not leak.get("enabled") else ("是" if leak.get("leak") else "否"),
            "elapsed": c.get("elapsed"),
            "human": (ratings.get(c["candidate_id"]) or {}).get("score"),
            "text": text,
            **m,
        })

    sims = []
    sim_lines = ["| | " + " | ".join(f"`{r['id']}`" for r in rows) + " |",
                 "|---|" + "---:|" * len(rows)]
    for a in rows:
        cells = []
        for b in rows:
            if a is b:
                cells.append("—")
            else:
                cells.append(f"{_sim(a['text'], b['text']):.3f}")
        sim_lines.append(f"| `{a['id']}` | " + " | ".join(cells) + " |")
    for a, b in itertools.combinations(rows, 2):
        sims.append(_sim(a['text'], b['text']))

    chars = [r["chars"] for r in rows]
    lines = [
        "# 多次试验横向报告",
        "",
        f"- run: `{base.name}`",
        f"- 候选数: {len(rows)}",
        f"- task: `{review.get('task', {}).get('title', '')}`",
        "",
        "## 结构 / 自动指标",
        "",
        "| 候选 | 字符 | 标题 | 阶段 | 3.1 行 | 3.4 行 | ASCII 行 | Mermaid 行 | 自动 | 泄漏 | 耗时 | 人工 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['id']}` | {r['chars']} | {r['headings']} | {r['stages']} | "
            f"{r['t31_rows']} | {r['t34_rows']} | {r['ascii_lines']} | {r['mermaid_lines']} | "
            f"{r['auto']} | {r['leak']} | {r['elapsed']}s | {r['human'] if r['human'] is not None else '-'} |"
        )
    lines += [
        "",
        "## 长度与稳定性",
        "",
        f"- 平均字符：{statistics.mean(chars):.1f}",
        f"- 范围：{min(chars)} ~ {max(chars)}",
        f"- 平均两两相似度：{statistics.mean(sims):.3f}" if sims else "- 平均两两相似度：—",
        f"- 相似度范围：{min(sims):.3f} ~ {max(sims):.3f}" if sims else "",
        "",
        "相似度高说明输出稳定但可能同质；低说明随机波动大，需要更多人工复核。",
        "",
        "## 两两相似度",
        "",
        *sim_lines,
        "",
        "## 怎么用",
        "",
        "```bash",
        f"python3 -m evolve rate --run {base} --candidate <id> --score 1-5",
        f"python3 -m evolve select --run {base} --by human",
        "```",
        "",
    ]
    out = base / "TRIALS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
