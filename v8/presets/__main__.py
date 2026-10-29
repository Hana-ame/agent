"""CLI：对一个会话挂 N 条预设，全部并行检查，聚合结果。

用法（在 v8/ 下）：
    python -m presets --presets presets/examples/presets \
        --conv presets/examples/conv_example.json --state mainline
    python -m presets --presets ... --conv ... --state full --dry   # 只看生成的 DAG
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

import dag8  # noqa: E402

from presets.builder import build_dag  # noqa: E402
from presets.presets import load_presets  # noqa: E402
from presets.state import load_conv, render_state  # noqa: E402


def human_summary(summary: dict) -> str:
    lines = [f"检查 {summary.get('checked', 0)} 条预设，命中 {summary.get('satisfied', 0)} 条："]
    for row in summary.get("hits", []):
        lines.append(f"  [HIT] {row['preset']}  score={row.get('score')}  {row.get('reason','')}")
    for row in summary.get("all", []):
        if not row.get("satisfied"):
            lines.append(f"  [ok ] {row['preset']}  score={row.get('score')}")
    return "\n".join(lines)


class _MockAgent:
    """离线假 agent：按提示词里的关键词返回固定 JSON，不调模型。"""

    def run(self, prompt: str, **_kw) -> str:
        p = (prompt or "")[:3000]
        if "澄清" in p:
            return json.dumps({"satisfied": True, "score": 4,
                               "reason": "用户表达困惑且助手未举例澄清（mock）"},
                              ensure_ascii=False)
        if "捏造" in p:
            return json.dumps({"satisfied": False, "score": 1,
                               "reason": "助手只转述已有材料（mock）"},
                              ensure_ascii=False)
        if "主轴" in p:
            return json.dumps({"satisfied": False, "score": 1,
                               "reason": "对话仍在主题内（mock）"},
                              ensure_ascii=False)
        return json.dumps({"satisfied": False, "score": 0,
                           "reason": "mock 默认未命中"}, ensure_ascii=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="presets", description="预设检查 DAG")
    ap.add_argument("--presets", required=True, help="存有预设 *.json 的目录")
    ap.add_argument("--conv", required=True, help="会话 JSON")
    ap.add_argument("--state", default="mainline", choices=["full", "last", "mainline"])
    ap.add_argument("--last", type=int, default=1, help="state=last 时取最近几条")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=0, help="0=按预设数")
    ap.add_argument("--rpm", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=600)
    ap.add_argument("--dry", action="store_true", help="只生成 DAG 不跑模型")
    ap.add_argument("--mock", action="store_true", help="用假 agent 跑通全流程，不调模型")
    ap.add_argument("--timeout", type=int, default=300, help="每个 pi 节点超时（秒）")
    ap.add_argument("--db", default=None, help="dag8 sqlite 路径")
    ap.add_argument("--out", default=None, help="run 目录")
    args = ap.parse_args(argv)

    presets = load_presets(args.presets)
    conv = load_conv(args.conv)
    state_text = render_state(conv, mode=args.state, last_n=args.last)
    pi = {"timeout": args.timeout}
    if args.provider:
        pi["provider"] = args.provider
    if args.model:
        pi["model"] = args.model
    cfg = build_dag(presets, pi=pi or None,
                    name=f"preset_check_{Path(args.conv).stem}")

    if args.dry:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        print(f"\n# 状态文本（{args.state}，{len(state_text)} 字符）\n{state_text}", file=sys.stderr)
        return 0

    run_dir = Path(args.out or (V8 / "data" / "presets" / "runs"
                                / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])))
    run_dir.mkdir(parents=True, exist_ok=True)
    dag_path = run_dir / "dag.json"
    dag_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "state.txt").write_text(state_text, encoding="utf-8")

    res = dag8.run(
        str(dag_path), seed_content=state_text,
        db_path=args.db or (run_dir / "run.sqlite3"),
        max_workers=args.workers or len(presets), rpm=args.rpm,
        use_cache=False, max_seconds=args.max_seconds,
        agent=_MockAgent() if args.mock else None,
    )
    if res.status != "ok" or not res.result:
        print(f"DAG 失败：{res.status}\n{res.summary()}", file=sys.stderr)
        return 1
    try:
        summary = json.loads(res.result)
    except ValueError:
        summary = {"raw": res.result}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n" + human_summary(summary))
    print(f"\nrun 目录：{run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
