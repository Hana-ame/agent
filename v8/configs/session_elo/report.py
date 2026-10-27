#!/usr/bin/env python3
"""生成 ELO 排行 Markdown / JSON。"""
from __future__ import annotations
import json, os, time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]

def cell(x, n=70):
    s = str(x or "").replace("|", "\\|").replace("\n", " ").strip()
    return s[:n] + ("…" if len(s) > n else "")

def main() -> int:
    data = json.loads(os.environ["DAG8_INPUTS"])
    raw = str(data["ranking"]).strip()
    meta = json.loads(raw) if raw.startswith("{") else {"path": raw}
    ranking = json.loads(Path(meta["path"]).read_text(encoding="utf-8"))
    items = ranking["items"]
    for x in items:
        x["chars"] = len(x.pop("conclusion", "") or "")
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "session_elo_ranking.json").write_text(
        json.dumps(ranking, ensure_ascii=False, indent=2), encoding="utf-8")
    from pathlib import Path as _P
    total = len(json.loads((_P.home() / ".dsh" / "session-manager" / "state.json").read_text(encoding="utf-8"))["byId"])
    skipped = total - len(items)
    kinds = Counter(x.get("kind") or "unknown" for x in items)
    cwds = Counter(x.get("cwd") or "" for x in items)
    lines = [
        "# DSH Session ELO 排行", "",
        f"> 生成：{time.strftime('%Y-%m-%d %H:%M')}　样本：{len(items)} 个 session（有最终 assistant 结论）",
        "", "## 方法", "",
        "- 每个 session 取最后一个 assistant text 作为结论",
        "- 确定性质量分：内容量/具体性 30%、与用户诉求匹配 25%、结构 20%、行动性 15%、完成状态 10%",
        "- 质量分两两对局 → Elo（初始 1500，K=16，3 轮全对局）",
        "- **这是启发式排序，不是 LLM 语义评判**；ELO 只在本语料内可比",
        "", "## 样本", "",
        f"- state 中 session 总数：{total}",
        f"- 可评价（有最终 assistant 文本）：{len(items)}；跳过 {skipped}（无 transcript/结论）",
        "- 按类型：" + "、".join(f"{k} {v}" for k, v in kinds.most_common()),
        "- 工作区 Top：" + "、".join(f"{cell(k,40)} {v}" for k, v in cwds.most_common(8)),
        "", "## Top 30", "",
        "| # | Elo | 质量分 | Session | 标题 | 类型 | turns/tools | 字数 |",
        "|---:|---:|---:|---|---|---|---:|---:|",
    ]
    for x in items[:30]:
        lines.append(f"| {x['rank']} | {x['elo']:.0f} | {x['score']:.2f} | "
                     f"`{x['session_id'][:24]}` | {cell(x.get('title'), 40)} | "
                     f"{x.get('kind') or ''} | {x.get('turns')}/{x.get('tools')} | "
                     f"{x.get('chars', 0)} |")
    lines += ["", "## Top 15：top-level（人开的会话）", "",
              "| # | 总排名 | Elo | 质量分 | Session | 标题 | 字数 |",
              "|---:|---:|---:|---:|---|---|---:|"]
    for x in [x for x in items if x.get("kind") == "top-level"][:15]:
        lines.append(f"| {len([y for y in items if y.get('kind') == 'top-level' and y['rank'] <= x['rank']])} | {x['rank']} | {x['elo']:.0f} | "
                     f"{x['score']:.2f} | `{x['session_id'][:24]}` | {cell(x.get('title'), 40)} | {x.get('chars', 0)} |")
    lines += ["", "## Top 10：subagent", "",
              "| 总排名 | Elo | 质量分 | Session | 标题 | 字数 |",
              "|---:|---:|---:|---|---|---:|"]
    for x in [x for x in items if x.get("kind") == "subagent"][:10]:
        lines.append(f"| {x['rank']} | {x['elo']:.0f} | {x['score']:.2f} | "
                     f"`{x['session_id'][:24]}` | {cell(x.get('title'), 40)} | {x.get('chars', 0)} |")
    lines += ["", "## Bottom 10", "",
              "| # | Elo | 质量分 | Session | 标题 | 字数 |",
              "|---:|---:|---:|---|---|---:|"]
    for x in items[-10:]:
        lines.append(f"| {x['rank']} | {x['elo']:.0f} | {x['score']:.2f} | "
                     f"`{x['session_id'][:24]}` | {cell(x.get('title'), 40)} | "
                     f"{x.get('chars', 0)} |")
    lines += ["", "## 注意", "",
              "- Top 30 混合了 subagent 和 top-level；只看人开的会话请看 Top 15 top-level",
              "- Elo 绝对值会超过常见 0–3000（3 轮全对局累积），只看相对名次，不要当绝对分",
              "- 结论为空/极短的 session 会被压低；这是数据缺失，不一定是回答差",
              "- 启发式偏好长、有代码/路径/数字、结构清晰的回答，存在长度偏置",
              "- 要语义级评判，需要网关恢复后用 LLM judge 重排", ""]
    md = reports / "session_elo_ranking.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "path": str(md), "count": len(items),
                      "top": items[0]["session_id"] if items else ""}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
