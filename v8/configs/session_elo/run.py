#!/usr/bin/env python3
"""入口：全量 DSH session 质量评分 + ELO 排名。

python3 v8/configs/session_elo/run.py --limit 50   # 先抽样
python3 v8/configs/session_elo/run.py              # 全部
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "v8")):
    if p not in sys.path:
        sys.path.insert(0, p)
import dag8  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = HERE / "dag.json"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=3600)
    args = ap.parse_args()
    os.environ["DSH_ELO_LIMIT"] = str(args.limit)
    res = dag8.run(
        str(CFG), seed_content="",
        db_path=str(ROOT / "v8" / "data" / "session_elo" / "run.sqlite3"),
        max_workers=args.workers, rpm=None, use_cache=False,
        max_seconds=args.max_seconds,
    )
    print(res.summary())
    if res.result:
        try:
            print(json.dumps(json.loads(res.result), ensure_ascii=False, indent=2))
        except ValueError:
            print(res.result)
    return 0 if res.status == "ok" else 1

if __name__ == "__main__":
    raise SystemExit(main())
