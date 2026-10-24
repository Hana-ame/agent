#!/usr/bin/env python3
"""用 10 份示例校准 judge 默认阈值（7）。

做法：用 min_score=10 的 probe 配置强制失败，从 error 里读回分数；
再按 score>=7 推断默认阈值的通过率。

    python scripts/judge_calibration.py
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(V8 / "data" / "judge_calibration" / "samples"))
    ap.add_argument("--config", default=str(V8 / "configs" / "live" / "judge_score_probe.json"))
    ap.add_argument("--db", default=str(V8 / "data" / "judge_calibration.sqlite3"))
    ap.add_argument("--out", default=str(V8 / "data" / "judge_calibration.json"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rpm", type=int, default=2)
    args = ap.parse_args()

    samples = sorted(Path(args.samples).glob("*.md"))
    if not samples:
        print("没有样本")
        return 2

    rows = []
    for p in samples:
        text = p.read_text(encoding="utf-8")
        cmd = [sys.executable, "-m", "dag8", "run", args.config,
               "--seed", text, "--db", args.db,
               "--workers", str(args.workers), "--rpm", str(args.rpm),
               "--max-seconds", "300", "--json"]
        proc = subprocess.run(cmd, cwd=V8, capture_output=True, text=True, timeout=360)
        score = None
        status = "?"
        err = ""
        try:
            data = json.loads(proc.stdout)
            status = data.get("status", "?")
            err = "\n".join(e.get("error") or "" for e in data.get("edges", []))
        except ValueError:
            err = (proc.stderr or proc.stdout or "")[:300]
        if status == "ok":
            score = 10.0
        else:
            m = re.search(r"judge\s*分数\s*([0-9.]+)", err)
            if m:
                score = float(m.group(1))
        rows.append({"file": p.name, "status": status, "score": score,
                     "error": err[:160]})
        print(f"{p.name:24s} status={status:7s} score={score}", flush=True)

    scored = [r["score"] for r in rows if r["score"] is not None]
    pass7 = [s for s in scored if s >= 7]
    result = {
        "n": len(rows), "scored": len(scored),
        "mean": round(sum(scored) / len(scored), 2) if scored else None,
        "pass_default_7": len(pass7),
        "fail_default_7": len(scored) - len(pass7),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nscored {len(scored)}/{len(rows)}; mean {result['mean']}; "
          f"默认阈值 7 通过 {len(pass7)} / 失败 {len(scored)-len(pass7)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
