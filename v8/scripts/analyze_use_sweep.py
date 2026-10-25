#!/usr/bin/env python3
"""分析 use_sweep / use_sweep_sqlite：中位数、P95、CV、bootstrap 比率 CI。"""
from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent


def pct(xs, q):
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def boot_ratio(a, b, n=20000, seed=42):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        aa = [rng.choice(a) for _ in a]
        bb = [rng.choice(b) for _ in b]
        out.append(st.median(aa) / st.median(bb))
    return pct(out, .025), pct(out, .975)


def summ(xs):
    lat = [r["wall"] for r in xs if r["status"] == "ok"]
    q = [r.get("quality", r.get("q", 0)) for r in xs]
    return {"n": len(xs), "ok": len(lat), "quality": round(st.mean(q), 2),
            "p50": round(st.median(lat), 2), "p95": round(pct(lat, .95), 2),
            "cv": round(st.pstdev(lat) / st.mean(lat), 2)}


def main() -> int:
    rows = json.loads((V8 / "data" / "use_sweep.json").read_text(encoding="utf-8"))
    sql = json.loads((V8 / "data" / "use_sweep_sqlite.json").read_text(encoding="utf-8"))
    out = {"groups": {}, "ratios": {}}

    for dag in ("research_brief", "sqlite_report", "script_judge"):
        for model in ("deepseek-flash", "deepseek-v4-flash"):
            xs = [r for r in rows if r["dag"] == dag and r["model"] == model]
            out["groups"][f"{dag}/{model}"] = summ(xs)

    for model in ("deepseek-flash", "deepseek-v4-flash"):
        xs = [r for r in sql if r["variant"] == "base" and r["model"] == model]
        out["groups"][f"sqlite_base/{model}"] = summ(xs)

    flash = ([r["wall"] for r in rows if r["dag"] == "sqlite_report"
              and r["model"] == "deepseek-flash"]
             + [r["wall"] for r in sql if r["variant"] == "base"
                and r["model"] == "deepseek-flash"])
    v4 = ([r["wall"] for r in rows if r["dag"] == "sqlite_report"
           and r["model"] == "deepseek-v4-flash"]
          + [r["wall"] for r in sql if r["variant"] == "base"
             and r["model"] == "deepseek-v4-flash"])
    lo, hi = boot_ratio(flash, v4)
    out["ratios"]["sqlite_combined_flash_vs_v4"] = {
        "n_flash": len(flash), "n_v4": len(v4),
        "p50_flash": round(st.median(flash), 2),
        "p50_v4": round(st.median(v4), 2),
        "ratio": round(st.median(flash) / st.median(v4), 2),
        "ci95": [round(lo, 2), round(hi, 2)],
        "speedup_ci95": [round(1 / hi, 1), round(1 / lo, 1)],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    (V8 / "data" / "use_sweep_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
