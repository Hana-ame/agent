#!/usr/bin/env python3
"""从全量评分里选样本并生成对局。"""
from __future__ import annotations
import json, os, random, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def brief(x, max_chars=800):
    return {
        "session_id": x["session_id"], "title": x.get("title") or "",
        "kind": x.get("kind") or "", "score": x.get("score"),
        "chars": len(x.get("conclusion") or ""),
        "excerpt": (x.get("conclusion") or "")[:max_chars],
    }

def main() -> int:
    src = ROOT / "v8" / "data" / "session_elo" / "scored.json"
    items = json.loads(src.read_text(encoding="utf-8"))["items"]
    if not items:
        raise SystemExit("scored.json 为空")
    n = int(os.environ.get("ELO_JUDGE_N", "4"))  # 每组人数
    top = items[:n]
    bottom = items[-n:]
    mid_pool = items[n:-n] or items
    rnd = random.Random(20261028)
    mid = rnd.sample(mid_pool, min(n, len(mid_pool)))
    players = top + mid + bottom
    pairs = []
    seen = set()
    for i, x in enumerate(players):
        opponents = rnd.sample([j for j in range(len(players)) if j != i], min(3, len(players) - 1))
        for j in opponents:
            key = tuple(sorted((i, j)))
            if key in seen: continue
            seen.add(key)
            pairs.append({"pair_id": f"match_{len(pairs)+1}", "a": brief(players[key[0]]), "b": brief(players[key[1]])})
    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "pairs": pairs}
    path = ROOT / "v8" / "data" / "session_elo_judge" / "pairs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"path": str(path), "count": len(pairs), "pairs": pairs}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
