"""python -m dag8 的命令行入口。

    python -m dag8 run configs/demo.json                 # 用配置里的 seed
    python -m dag8 run configs/demo.json --seed "新的任务"  # 覆盖 seed
    python -m dag8 run configs/demo.json --dry-run       # 不调 pi，只验配置与占位符
    python -m dag8 show <run_id>                         # 看一次运行的全部内容
    python -m dag8 runs                                  # 列出最近的运行
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dag8 import run as run_dag
from dag8.store import Store


def _cmd_run(args: argparse.Namespace) -> int:
    seed = args.seed
    if args.seed_file:
        seed = Path(args.seed_file).read_text(encoding="utf-8")
    if args.check_model:
        return _check_model(args.config)
    res = run_dag(args.config, seed_content=seed, db_path=args.db, dry_run=args.dry_run)
    print(res.summary())
    print(f"\ndb: {res.db_path}")
    if res.result:
        print("\n--- result ---")
        print(res.result)
    return 0 if res.status == "ok" else 1


def _check_model(config_path: str) -> int:
    """只验 provider/model 真能跑，不跑边。

    必要性：`pi` 对未注册的模型只打一行 Warning 就按 custom model id 透传，
    退出码仍是 0。模型名拼错时 v8 看不出异常，只能等跑到一半才炸。
    这里先发一次最小请求，让问题在开跑之前暴露。
    """
    from dag8.config import load as load_config
    from dag8.pi import PiAgent, PiError

    cfg = load_config(config_path)
    # 预检只试一次：坏模型的重试代价是 timeout×retries（实测 240s×3 要三分半），
    # 而预检的目的就是尽快暴露问题，不值得为它付重试的钱。
    cfg.pi.retries = 1
    agent = PiAgent(cfg.pi)
    target = f"{cfg.pi.provider}/{cfg.pi.model}"
    print(f"检查 {target} ...", flush=True)
    try:
        out = agent.run("reply with the single word ok", timeout=min(cfg.pi.timeout, 60))
    except PiError as exc:
        print(f"✗ {target} 不可用：{exc}")
        print("\n可用模型见 v8/MODELS.md")
        return 1
    print(f"✓ {target} 可用（返回 {len(out)} 字符）")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        info = store.run_info(args.run_id)
        print(json.dumps({k: info[k] for k in
                          ("run_id", "dag_name", "status", "started_at", "finished_at",
                           "result_tag", "error")}, ensure_ascii=False, indent=2))
        print("\n--- edges ---")
        for e in store.edge_records(args.run_id):
            print(f"  [{e['layer']}] {e['edge_id']}  {e['status']}  "
                  f"{e['model'] or '-'}  {e['elapsed'] or 0:.1f}s"
                  + (f"  ERR {e['error'][:100]}" if e["error"] else ""))
        print("\n--- contents ---")
        for it in store.all_contents(args.run_id):
            tags = ",".join(it["tags"])
            body = it["content"]
            print(f"  #{it['id']} gen={it['gen']} tags=[{tags}] via {it['source_edge'] or 'seed'}"
                  f"  {len(body)} 字符")
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        rows = store.conn.execute(
            "SELECT run_id, dag_name, status, started_at, "
            "(SELECT COUNT(*) FROM contents c WHERE c.run_id=r.run_id) AS n "
            "FROM runs r ORDER BY started_at DESC LIMIT ?",
            (args.limit,),
        ).fetchall()
        if not rows:
            print("还没有任何运行记录")
            return 0
        print(f"{'run_id':14} {'dag':20} {'status':10} {'contents':>8}")
        for r in rows:
            print(f"{r['run_id']:14} {r['dag_name'][:20]:20} {r['status']:10} {r['n']:>8}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dag8", description="JSON 配置 + pi agent + sqlite3")
    ap.add_argument("--db", default=None, help="sqlite3 路径（默认 v8/data/dag8.sqlite3）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("run", help="跑一张配置好的图")
    rp.add_argument("config")
    rp.add_argument("--seed", default=None, help="覆盖配置里的 seed_content")
    rp.add_argument("--seed-file", default=None, help="从文件读 seed（长文本用，优先级高于 --seed）")
    rp.add_argument("--dry-run", action="store_true", help="不调 pi，把渲染后的 prompt 当输出")
    rp.add_argument("--check-model", action="store_true",
                    help="只验证 provider/model 能否真的调通，不跑边")
    rp.set_defaults(func=_cmd_run)

    sp = sub.add_parser("show", help="查看一次运行")
    sp.add_argument("run_id")
    sp.set_defaults(func=_cmd_show)

    lp = sub.add_parser("runs", help="列出最近的运行")
    lp.add_argument("--limit", type=int, default=20)
    lp.set_defaults(func=_cmd_runs)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
