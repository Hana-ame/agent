"""命令行：python3 -m dags list | run <name> [-o OUTPUT]"""
from __future__ import annotations

import argparse
import sys

from dags import list_dags, print_timings, run_dag


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("dags", description="可扩展 DAG 库")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出所有已注册 DAG")

    rp = sub.add_parser("run", help="单次运行一个 DAG")
    rp.add_argument("name", help="DAG 名称（见 list）")
    rp.add_argument("-o", "--output", default=None, help="输出文件路径")
    rp.add_argument("--max-steps", type=int, default=30)

    args = ap.parse_args(argv)

    if args.cmd == "list":
        specs = list_dags()
        if not specs:
            print("(空)")
            return 0
        width = max(len(s.name) for s in specs)
        for s in specs:
            print(f"{s.name:<{width}}  {s.description}")
        return 0

    if args.cmd == "run":
        res = run_dag(args.name, output=args.output, max_steps=args.max_steps)
        print(f"\n=== {res['name']} ===")
        print(f"输出：{res['output']}")
        print(f"行数：{len(str(res['report']).splitlines())}")
        print_timings(res["timings"], res["elapsed"])  # type: ignore[arg-type]
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
