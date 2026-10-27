#!/usr/bin/env python3
"""用两两对局把质量分转成 Elo 排名。"""
from __future__ import annotations
import json, os, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
K = 16.0
ROUNDS = 3
EPS = 0.01

def main() -> int:
    data = json.loads(os.environ["DAG8_INPUTS"])
    scored = json.loads(data["scored"])
    items = json.loads(Path(scored["path"]).read_text(encoding="utf-8"))["items"]
    players = [x for x in items if x.get("evaluable") and (x.get("conclusion") or "").strip()]
    n = len(players)
    elo = [1500.0] * n
    wins = [0] * n; losses = [0] * n; draws = [0] * n; matches = [0] * n
    order = list(range(n))
    for rnd in range(ROUNDS):
        random.Random(20261027 + rnd).shuffle(order)
        # 用打乱后的索引做两两循环，避免固定顺序偏差
        for a_pos in range(n):
            i = order[a_pos]
            si = players[i]["score"]
            ri = elo[i]
            for b_pos in range(a_pos + 1, n):
                j = order[b_pos]
                sj = players[j]["score"]
                d = si - sj
                outcome = 0.5 if abs(d) <= EPS else (1.0 if d > 0 else 0.0)
                exp_i = 1.0 / (1.0 + 10 ** ((elo[j] - ri) / 400.0))
                elo[i] = ri + K * (outcome - exp_i)
                elo[j] = elo[j] + K * ((1.0 - outcome) - (1.0 - exp_i))
                ri = elo[i]
                matches[i] += 1; matches[j] += 1
                if outcome == 1.0: wins[i] += 1; losses[j] += 1
                elif outcome == 0.0: wins[j] += 1; losses[i] += 1
                else: draws[i] += 1; draws[j] += 1
    ranked = []
    for idx, p in enumerate(players):
        q = {k: p.get(k) for k in ("session_id", "title", "kind", "cwd", "updated_iso",
                                    "turns", "tools", "status", "conclusion", "score", "components")}
        q.update({"elo": round(elo[idx], 1), "matches": matches[idx],
                  "wins": wins[idx], "losses": losses[idx], "draws": draws[idx]})
        ranked.append(q)
    ranked.sort(key=lambda x: (-x["elo"], -x["score"]))
    for rank, x in enumerate(ranked, 1):
        x["rank"] = rank
    out = ROOT / "v8" / "data" / "session_elo" / "ranking.json"
    out.write_text(json.dumps({"count": len(ranked), "rounds": ROUNDS, "k": K,
                               "items": ranked}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"path": str(out), "count": len(ranked),
                      "top": [(x["session_id"][:20], x["elo"], x["score"]) for x in ranked[:5]]},
                     ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
