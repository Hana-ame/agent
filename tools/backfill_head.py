#!/usr/bin/env python
"""给**没有 HEAD.json 的 meta run** 追溯补写 git 式回档链。

`tools/meta_dag.py` 是在跑完 s12 之后才引入 `write_head()` 的，所以 s12
（以及改动生效前就在跑的 s13）各代目录里没有 HEAD.json，
`tools/boot_rollback.py` 看不到它们。这个工具按该 run 的 `RESULT.json`
（每代的 demo / bootstrap 验收结果）+ 磁盘上的 `vea_v6/` 文件重建链：

- gen1 的 `parent` 置 `null`（这些 run 没留 gen0 快照，无法与磁盘现状挂钩）；
- genN（N>1）的 `parent` = 上一代的 `combined_sha256`；
- `byte_identical` 由 parent 自动判定；
- 只有 RESULT.json 里 demo 与 bootstrap **都通过**的代才入链。

用法：
    python tools/backfill_head.py meta/s12
    python tools/backfill_head.py meta/s13
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.meta_dag import write_head  # noqa: E402

RUNS = ROOT / "bootstrap_runs" / "meta"


def backfill(run_dir: Path) -> List[Dict[str, Any]]:
    result_path = run_dir / "RESULT.json"
    if not result_path.exists():
        raise SystemExit(f"{run_dir} 里没有 RESULT.json")
    records = json.loads(result_path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    prev: str = ""
    for rec in records:
        # 只把通过验收的代入链（与 meta_dag.py 主循环的写 HEAD 时机一致）
        if not (rec.get("demo") and rec.get("bootstrap")):
            print(f"  跳过 gen{rec.get('gen')}：未同时过自选任务与自举闸门")
            continue
        gdir = run_dir / f"gen{rec['gen']}"
        if not gdir.is_dir():
            print(f"  跳过 gen{rec.get('gen')}：目录不存在")
            continue
        head = write_head(gdir, {
            "gen": rec["gen"],
            "parent": prev or None,
            "mode": f"meta-dag:{rec.get('model_key', 'laptop-6.8')}",
            "model_key": rec.get("model_key", "laptop-6.8"),
            "self_verify": rec["bootstrap"],
            "self_verify_reason": rec.get("bootstrap_out", ""),
            "demo": rec["demo"],
            "demo_out": rec.get("demo_out", ""),
            "steps": sorted(p.name for p in (gdir / "steps").iterdir())
                     if (gdir / "steps").exists() else [],
            "note": "HEAD.json 为追溯补写：该 run 跑于 write_head 引入之前，"
                    "gen1 无 gen0 快照可比，parent 置 null",
        })
        prev = head["combined_sha256"]
        out.append(head)
        print(f"  gen{rec['gen']}  {head['combined_sha256'][:16]}  "
              f"{head['total_bytes']}B  字节{'一致' if head.get('byte_identical') else '不同'}")
    return out


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description="给老 meta run 追溯补写 HEAD.json 回档链")
    ap.add_argument("run", help="run 目录名（相对 bootstrap_runs/meta/），如 s12")
    a = ap.parse_args(argv)

    run_dir = RUNS / a.run
    if not run_dir.is_dir():
        raise SystemExit(f"没有这个 run: {run_dir}")
    print(f"补写 {run_dir.relative_to(ROOT)}")
    heads = backfill(run_dir)
    print(f"\n完成：{len(heads)} 代入链。复核：python tools/boot_rollback.py verify meta/{a.run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
