#!/usr/bin/env python3
"""ELO 裁判 DAG 入口。"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "v8")):
    if p not in sys.path: sys.path.insert(0, p)
import dag8  # noqa: E402
from v7.llm import OpenAICompatibleClient  # noqa: E402

CFG = Path(__file__).resolve().parent / "dag.json"

class LocalJudge:
    def complete(self, prompt: str, **_kwargs) -> str:
        marker = prompt.find('"pairs"')
        if marker < 0: return json.dumps({"votes": []})
        start = prompt.rfind("{", 0, marker)
        pairs = json.loads(prompt[start:]).get("pairs", [])
        votes = []
        for p in pairs:
            a, b = p["a"], p["b"]
            sa, sb = a.get("score") or 0, b.get("score") or 0
            winner = "tie" if abs(sa - sb) <= 0.01 else ("A" if sa > sb else "B")
            votes.append({"pair_id": p["pair_id"], "winner": winner, "reason": "local score"})
        return json.dumps({"votes": votes}, ensure_ascii=False)

    def embed(self, text: str, **_kwargs): return [0.0] * 32

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("local", "http"), default="http")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=900)
    args = ap.parse_args()
    import os
    os.environ["ELO_JUDGE_N"] = str(args.n)
    os.environ["ELO_JUDGE_BACKEND"] = args.backend
    kwargs = {"graph_llm_client": LocalJudge()} if args.backend == "local" else \
             {"graph_llm_client": OpenAICompatibleClient(timeout=300, max_retries=5)}
    res = dag8.run(str(CFG), seed_content="", max_workers=args.workers,
                   db_path=str(ROOT / "v8" / "data" / "session_elo_judge" / "run.sqlite3"),
                   rpm=None if args.backend == "local" else 1,
                   use_cache=False, max_seconds=args.max_seconds, **kwargs)
    print(res.summary())
    if res.result:
        try: print(json.dumps(json.loads(res.result), ensure_ascii=False, indent=2))
        except ValueError: print(res.result)
    return 0 if res.status == "ok" else 1

if __name__ == "__main__": raise SystemExit(main())
