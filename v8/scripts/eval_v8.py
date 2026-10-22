"""跑同一张配置 N 次，统计 v8 的稳定性 / 速度 / 正确率。

和 v7 的 dag-eval 是同一套判据，只是数据源从内存换成 sqlite：

- 稳定性：输出结构字段的一致率、两次输出的 3-gram Jaccard、长度变异系数
- 速度：每条边的 min / median / max，以及每次 run 的总耗时
- 正确率：结构硬检查（非空 / 结尾完整 / 有标题 / 到达 result_tag）

    python v8/scripts/eval_v8.py v8/configs/demo.json --runs 3
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V8))

import dag8  # noqa: E402
from dag8.store import Store  # noqa: E402


def _structure(text: str) -> dict:
    lines = text.splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    return {
        "chars": len(text),
        "lines": len(lines),
        "sections": len(re.findall(r"^#{1,4} ", text, re.M)),
        "ends_clean": bool(re.search(r"[。！？.!)\]》”\"]\s*$", text.rstrip())),
        "has_error": any(m in text for m in ("Traceback", "Error:", "Exception:")),
    }


def _ngrams(t: str, n: int = 3) -> set:
    s = re.sub(r"\s+", "", t)
    return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _ngrams(a), _ngrams(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def _agreement(vals) -> float:
    return max(vals.count(v) for v in set(vals)) / len(vals) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--db", default=str(V8 / "data" / "eval.sqlite3"))
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--seed-file", default=None, help="配置里 seed_content 为空时，必须用它提供种子")
    args = ap.parse_args()

    seed = None
    if args.seed_file:
        seed = Path(args.seed_file).read_text(encoding="utf-8")

    run_ids, wall = [], []
    print(f"=== 跑 {args.runs} 次：{args.config} ===")
    for i in range(args.runs):
        t0 = time.perf_counter()
        res = dag8.run(args.config, seed_content=seed, db_path=args.db)
        dt = time.perf_counter() - t0
        wall.append(dt)
        run_ids.append(res.run_id)
        print(f"  run {i + 1}: {res.status}  {dt:.1f}s  {len(res.result or '')} 字符  "
              + " ".join(f"{e.edge_id}={e.elapsed:.1f}s" for e in res.edges))

    with Store(args.db) as st:
        reports = [st.result(rid, st.run_info(rid)["result_tag"]) or "" for rid in run_ids]
        edge_rows = [st.edge_records(rid) for rid in run_ids]

    structs = [_structure(r) for r in reports]
    fields = ["chars", "lines", "sections", "ends_clean", "has_error"]
    agreement = {f: round(_agreement([s[f] for s in structs]), 3) for f in fields}

    pairs = [_jaccard(reports[i], reports[j])
             for i in range(len(reports)) for j in range(i + 1, len(reports))]
    jac = round(statistics.fmean(pairs), 3) if pairs else 1.0

    lens = [len(r) for r in reports]
    cv = round(statistics.pstdev(lens) / statistics.fmean(lens), 3) if statistics.fmean(lens) else 0.0

    # 每条边的耗时分布
    per_edge: dict[str, list[float]] = {}
    for rows in edge_rows:
        for r in rows:
            if r["elapsed"]:
                per_edge.setdefault(r["edge_id"], []).append(r["elapsed"])
    edge_stats = {
        k: {"min": round(min(v), 1), "median": round(statistics.median(v), 1), "max": round(max(v), 1)}
        for k, v in per_edge.items()
    }

    # 判据来自配置的 eval 段：不同任务的期望结构不一样，不能一套检查套所有图
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ev = cfg.get("eval") or {}
    min_chars = int(ev.get("min_chars", 1))
    min_sections = int(ev.get("min_sections", 0))

    checks = {
        "non_empty": all(bool(r.strip()) for r in reports),
        f"min_chars>={min_chars}": all(len(r) >= min_chars for r in reports),
        "ends_clean": all(s["ends_clean"] for s in structs),
        "no_error": all(not s["has_error"] for s in structs),
        "all_edges_ok": all(r["status"] == "ok" for rows in edge_rows for r in rows),
    }
    if min_sections:
        checks[f"min_sections>={min_sections}"] = all(s["sections"] >= min_sections for s in structs)
    correct = round(sum(checks.values()) / len(checks), 3)

    lines = [
        f"# v8 评测 · {Path(args.config).stem}",
        "",
        f"- 运行次数：{args.runs}",
        f"- 端到端耗时：{', '.join(f'{w:.1f}s' for w in wall)}",
        "",
        "## 稳定性",
        "",
        "| 字段 | 一致率 |",
        "|---|--:|",
        *[f"| {k} | {v} |" for k, v in agreement.items()],
        "",
        f"- 输出 3-gram Jaccard 均值：**{jac}**",
        f"- 长度变异系数 CV：**{cv}**（各次 {lens}）",
        "",
        "## 速度",
        "",
        "| 边 | min | median | max |",
        "|---|--:|--:|--:|",
        *[f"| {k} | {v['min']} | {v['median']} | {v['max']} |" for k, v in edge_stats.items()],
        "",
        "## 正确率",
        "",
        "| 检查项 | 结果 |",
        "|---|:--:|",
        *[f"| {k} | {'✓' if v else '✗'} |" for k, v in checks.items()],
        "",
        f"综合：**{correct:.0%}**",
    ]
    text = "\n".join(lines)
    print("\n" + text)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\n写出 {args.output}")
    return 0 if correct == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
