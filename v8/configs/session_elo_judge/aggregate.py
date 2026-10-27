#!/usr/bin/env python3
"""用裁判票算 Elo 并生成报告。"""
from __future__ import annotations
import json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
K = 32.0
BASE = 1500.0

def main() -> int:
    data = json.loads(os.environ["DAG8_INPUTS"])
    pairs = json.loads(data["pairs"])["pairs"]
    votes = json.loads(data["votes"]).get("votes", [])
    vmap = {v.get("pair_id"): v for v in votes if isinstance(v, dict)}
    elo, matches, wins, losses, ties = {}, {}, {}, {}, {}
    for p in pairs:
        for side in ("a", "b"):
            sid = p[side]["session_id"]
            elo.setdefault(sid, BASE); matches.setdefault(sid, 0)
            wins.setdefault(sid, 0); losses.setdefault(sid, 0); ties.setdefault(sid, 0)
    details = []
    for p in pairs:
        a, b = p["a"], p["b"]
        sa, sb = a["session_id"], b["session_id"]
        v = vmap.get(p["pair_id"], {})
        winner = v.get("winner", "tie")
        if winner not in ("A", "B", "tie"): winner = "tie"
        ra, rb = elo[sa], elo[sb]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        score_a = 1.0 if winner == "A" else (0.0 if winner == "B" else 0.5)
        elo[sa] = ra + K * (score_a - ea)
        elo[sb] = rb + K * ((1.0 - score_a) - (1.0 - ea))
        matches[sa] += 1; matches[sb] += 1
        if winner == "A": wins[sa] += 1; losses[sb] += 1
        elif winner == "B": wins[sb] += 1; losses[sa] += 1
        else: ties[sa] += 1; ties[sb] += 1
        details.append({"pair_id": p["pair_id"], "a": sa, "b": sb, "winner": winner,
                        "reason": v.get("reason", "")})
    players = {}
    for p in pairs:
        for side in ("a", "b"):
            x = p[side]; sid = x["session_id"]
            players[sid] = {**x, "elo": round(elo[sid], 1), "matches": matches[sid],
                            "wins": wins[sid], "losses": losses[sid], "ties": ties[sid]}
    ranked = sorted(players.values(), key=lambda x: (-x["elo"], -(x.get("score") or 0)))
    for i, x in enumerate(ranked, 1): x["rank"] = i
    result = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "backend": os.environ.get("ELO_JUDGE_BACKEND", "unknown"),
              "players": len(ranked), "pairs": len(pairs), "votes": len(votes),
              "items": ranked, "details": details}
    reports = ROOT / "reports"; reports.mkdir(parents=True, exist_ok=True)
    (reports / "session_elo_judge.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# ELO 裁判 DAG 结果", "",
             f"> 生成：{result['generated_at']}　选手：{result['players']}　对局：{result['pairs']}　有效票：{result['votes']}", "",
             "## 方法", "",
             "- 样本：top / bottom / 随机三段",
             f"- 裁判后端：{os.environ.get('ELO_JUDGE_BACKEND', 'unknown')}",
             "- 裁判：逐对判 A/B/tie，票直接更新 Elo（K=32，初始 1500）",
             "- 这是裁判票的结果，不是启发式质量分", "",
             "## 排名", "",
             "| # | Elo | 质量分(参考) | Session | 标题 | 战绩 |",
             "|---:|---:|---:|---|---|---|"]
    for x in ranked:
        lines.append(f"| {x['rank']} | {x['elo']:.0f} | {x.get('score', 0):.2f} | "
                     f"`{x['session_id'][:24]}` | {str(x.get('title') or '')[:40]} | "
                     f"{x['wins']}W/{x['losses']}L/{x['ties']}T |")
    lines += ["", "## 裁判明细", "", "| 对局 | A | B | 胜者 | 理由 |", "|---|---|---|---|---|"]
    for d in details:
        lines.append(f"| {d['pair_id']} | `{d['a'][:18]}` | `{d['b'][:18]}` | {d['winner']} | {str(d['reason'])[:60]} |")
    md = reports / "session_elo_judge.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "path": str(md), "players": len(ranked),
                      "pairs": len(pairs), "votes": len(votes)}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
