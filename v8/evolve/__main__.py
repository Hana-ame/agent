"""python -m evolve 的 CLI。

    python -m evolve run --task evolve/tasks/troll_notes.json --population 3
    python -m evolve from-runs --db data/dag8.sqlite3 --last 3
    python -m evolve review --run data/human-eval/<dir>
    python -m evolve rate --run <dir> --candidate <id> --score 4 --note "..."
    python -m evolve select --run <dir> --by human
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from evolve.genome import Genome
from evolve.review import build_review, rate_candidate, render_table, select_candidate
from evolve.runner import run_candidate
from evolve.tasks import Task
from evolve.from_runs import build_from_runs


class _MockAgent:
    """离线自检用的假 agent；不调模型，只返回固定结构文本。"""

    def run(self, prompt: str, **kwargs: Any) -> str:
        return (
            "# Mock 输出\n\n"
            "## 一、 一句话核心概括（Executive Summary）\n\n"
            "这是 mock agent 生成的占位结论，用于验证评价包链路。\n\n"
            "## 二、 完整内容摘要（Comprehensive Abstract）\n\n"
            "1. **占位**：mock 输出，不调模型。\n\n"
            "## 三、 结构化深度阅读笔记（Structured Reading Notes）\n\n"
            "### 1. 案例对标：单一指标考核的系统性失灵\n\n"
            "| 对比维度 | A | B | C |\n|---|---|---|---|\n| 核心 | x | y | z |\n\n"
            "### 2. 核心管理学推演：SOP 是如何杀死组织的？\n\n"
            "1. **第一阶段**：x\n2. **第二阶段**：x\n3. **第三阶段**：x\n"
            "4. **第四阶段**：x\n5. **第五阶段**：x\n\n"
            "### 3. 好组织运作的“双支柱模型”\n\n"
            "* **支柱一**：x\n* **支柱二**：y\n\n"
            "### 4. 原文口语/谐音词汇对照勘误表（Transcript Glossary）\n\n"
            "| 原文识别词 | 实际对应词 / 典故 | 语境释义 |\n|---|---|---|\n"
            "| a | b | c |\n\n"
            "## 四、 树状思维导图（Text & Mermaid Mind Map）\n\n"
            "### 1. 结构化层级导图（ASCII Tree）\n\n```text\nroot\n└─ x\n```\n\n"
            "### 2. Mermaid 思维导图代码\n\n```mermaid\nmindmap\n  root((x))\n    a\n```\n"
        )


def _new_run_dir(task_id: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    p = V8 / "data" / "human-eval" / f"{stamp}-{task_id}"
    p.mkdir(parents=True, exist_ok=False)
    return p


def cmd_run(args: argparse.Namespace) -> int:
    task = Task.load(args.task)
    base = Genome.from_config(task.config, name="base", workers=args.workers)
    run_dir = Path(args.out) if args.out else _new_run_dir(task.id)
    run_dir.mkdir(parents=True, exist_ok=True)

    population: List[Genome] = [base]
    rng = random.Random(args.seed)
    ops = [x.strip() for x in args.ops.split(",") if x.strip()] if args.ops else None
    guard = 0
    while len(population) < max(1, args.population) and guard < args.population * 20:
        guard += 1
        child = base.mutate(rng, allowed=ops)
        if all(c.id != child.id for c in population):
            population.append(child)

    (run_dir / "task.json").write_text(
        json.dumps(task.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "genomes.json").write_text(
        json.dumps([g.to_dict() for g in population], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    agent = _MockAgent() if args.mock else None
    records = []
    for i, g in enumerate(population, 1):
        print(f"[{i}/{len(population)}] {g.id} {g.name} ...", flush=True)
        rec = run_candidate(
            g, task, run_dir, agent=agent, use_cache=not args.no_cache,
            source_path=args.source or None, answer_path=args.answer or None,
        )
        leak = rec.get("leakage") or {}
        print(f"    {rec['status']}  {rec['output_chars']} 字符  {rec['elapsed']}s"
              + (f"  泄漏×0.1" if leak.get("leak") else "")
              + (f"  ERR {rec['error']}" if rec.get("error") else ""), flush=True)
        records.append(rec)

    review_path = build_review(run_dir, task, records)
    print(f"\n评价包：{review_path}")
    print(f"评分表：{run_dir / 'ratings.json'}")
    print("\n" + render_table(run_dir))
    print(f"\n下一步：python -m evolve rate --run {run_dir} --candidate <id> --score 1-5")
    return 0 if any(r["status"] == "ok" for r in records) else 1


def cmd_from_runs(args: argparse.Namespace) -> int:
    run_dir = Path(args.out) if args.out else _new_run_dir("from-runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    path = build_from_runs(
        args.db, run_dir, last=args.last, title=args.title,
        source_path=args.source or None, answer_path=args.answer or None,
    )
    print(f"评价包：{path}")
    print(f"评分表：{run_dir / 'ratings.json'}")
    print("\n" + render_table(run_dir))
    print(f"\n下一步：python -m evolve rate --run {run_dir} --candidate <id> --score 1-5")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    review = run_dir / "review.md"
    if not review.is_file():
        print(f"没有评价包：{review}", file=sys.stderr)
        return 1
    print(review.read_text(encoding="utf-8"))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    if not (run_dir / "review.json").is_file():
        print(f"没有 review.json：{run_dir}", file=sys.stderr)
        return 1
    print(render_table(run_dir))
    return 0


def cmd_rate(args: argparse.Namespace) -> int:
    rate_candidate(args.run, args.candidate, args.score, args.note or "")
    print(f"已评分：{args.candidate} = {args.score}/5"
          + (f"  {args.note}" if args.note else ""))
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    result = select_candidate(args.run, by=args.by)
    print(f"selection：{Path(args.run) / 'selection.json'}")
    print(f"winner：{result['winner']}（{result['by']}）")
    for r in result["ranking"]:
        print(f"  {r['score']:>6}  human={r['human']} auto={r['auto']}  {r['candidate_id']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="evolve", description="受约束 DAG 演化 + 人工评价")
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("run", help="生成候选并跑出人工评价包")
    rp.add_argument("--task", required=True, help="task JSON")
    rp.add_argument("--population", type=int, default=3)
    rp.add_argument("--seed", type=int, default=7)
    rp.add_argument("--workers", type=int, default=1)
    rp.add_argument("--ops", default="", help="逗号分隔的变异算子")
    rp.add_argument("--out", default=None)
    rp.add_argument("--no-cache", action="store_true")
    rp.add_argument("--mock", action="store_true", help="用假 agent，不调模型")
    rp.add_argument("--source", default=None, help="源材料文本/JSON（泄漏审计用）")
    rp.add_argument("--answer", default=None, help="参考答案（泄漏审计用）")
    rp.set_defaults(func=cmd_run)

    fp = sub.add_parser("from-runs", help="从历史 run 直接生成评价包")
    fp.add_argument("--db", default=str(V8 / "data" / "dag8.sqlite3"))
    fp.add_argument("--last", type=int, default=3)
    fp.add_argument("--out", default=None)
    fp.add_argument("--title", default="历史 run 回顾")
    fp.add_argument("--source", default=None, help="源材料文本/JSON（泄漏审计用）")
    fp.add_argument("--answer", default=None, help="参考答案（泄漏审计用）")
    fp.set_defaults(func=cmd_from_runs)

    vp = sub.add_parser("review", help="打印 review.md")
    vp.add_argument("--run", required=True)
    vp.set_defaults(func=cmd_review)

    lp = sub.add_parser("list", help="打印评分表")
    lp.add_argument("--run", required=True)
    lp.set_defaults(func=cmd_list)

    rp2 = sub.add_parser("rate", help="给候选打 1~5 分")
    rp2.add_argument("--run", required=True)
    rp2.add_argument("--candidate", required=True)
    rp2.add_argument("--score", type=int, required=True)
    rp2.add_argument("--note", default="")
    rp2.set_defaults(func=cmd_rate)

    sp = sub.add_parser("select", help="按人工/自动/混合选候选")
    sp.add_argument("--run", required=True)
    sp.add_argument("--by", default="human", choices=["human", "auto", "combined"])
    sp.set_defaults(func=cmd_select)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
