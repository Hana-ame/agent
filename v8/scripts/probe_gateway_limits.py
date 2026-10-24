#!/usr/bin/env python3
"""探测网关并发/RPM 阈值：小请求、无重试，统计 429 与延迟。

```bash
python scripts/probe_gateway_limits.py --rounds 1
```

结果写到 v8/data/gateway_limits.json。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

V8 = Path(__file__).resolve().parent.parent
BASE = "https://laptop-8000.moonchan.xyz/v1"
MODEL = "deepseek-flash"


def one(timeout: float = 60.0) -> tuple[int, float]:
    t0 = time.perf_counter()
    try:
        r = requests.post(
            BASE + "/chat/completions",
            headers={"Authorization": "Bearer not-needed"},
            json={"model": MODEL,
                  "messages": [{"role": "user", "content": "只回复：ok"}],
                  "max_tokens": 8, "temperature": 0},
            timeout=timeout,
        )
        return r.status_code, time.perf_counter() - t0
    except Exception:  # noqa: BLE001
        return -1, time.perf_counter() - t0


def burst(n: int) -> dict:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(lambda _: one(), range(n)))
    wall = time.perf_counter() - t0
    codes = [c for c, _ in results]
    lat = [x for _, x in results]
    return {
        "n": n,
        "codes": {str(c): codes.count(c) for c in sorted(set(codes))},
        "wall": round(wall, 1),
        "p50": round(statistics.median(lat), 1),
        "max": round(max(lat), 1),
        "ok": codes.count(200),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--sizes", default="1,2,3,4,6")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]

    rows = []
    for i in range(args.rounds):
        for n in sizes:
            row = burst(n)
            row["round"] = i + 1
            rows.append(row)
            print(f"round {i+1} n={n}: ok={row['ok']}/{n} codes={row['codes']} "
                  f"wall={row['wall']}s p50={row['p50']}s", flush=True)

    out = V8 / "data" / "gateway_limits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
