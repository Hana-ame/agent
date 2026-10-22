"""从 S1 157 板块抓取近 24 小时的 AI 相关讨论，导出成纯文本种子。

复用 v7 `scripts/s1/stage1st_ai.py` 的抓取与判定（那边已经处理了重试、
24h 回溯、LLM 分类），这里只负责把结果摊平成一份可喂给 v8 的文本。

    python v8/scripts/export_s1_seed.py -o /tmp/s1_posts.txt

产物结构：

    ## 帖子标题（链接）— 最后回复 2026-10-22 19:34，24h 回帖 30 条
    [2026-10-22 19:20] 回帖正文…
    [2026-10-22 19:24] 回帖正文…
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
ROOT = V8.parent
sys.path.insert(0, str(ROOT / "scripts" / "s1"))
sys.path.insert(0, str(ROOT))

import stage1st_ai as s1  # noqa: E402


def build_text(hours: int = 24) -> str:
    print("[1/3] 抓目录…")
    threads = s1.scrape_catalog()
    if not threads:
        raise SystemExit("目录抓取失败：0 帖")

    print(f"      {len(threads)} 帖，判定 AI 相关性…")
    ai = s1.classify_ai_threads(threads)
    print(f"      AI 相关 {len(ai)} 帖")

    since = dt.datetime.now() - dt.timedelta(hours=hours)
    catalog_active = sum(
        1 for t in threads
        if (lp := s1._parse_dt(t.get("last_post", ""))) is not None and lp >= since
    )

    print("[2/3] 回溯近 24 小时回帖…")
    data = s1.collect_24h(threads=ai, catalog_total=len(threads), catalog_active=catalog_active)
    print(f"      活跃 {data['active_total']} 帖，回帖 "
          f"{sum(it['posts_total_24h'] for it in data['threads'])} 条")

    print("[3/3] 摊平成文本…")
    out = [
        f"# S1 157（归墟）近 {hours} 小时 AI 相关讨论",
        "",
        f"- 采集时间：{data['generated_at']}",
        f"- 时间窗口：{data['window_start']} ~ {data['generated_at']}",
        f"- 板块统计：目录 {data['catalog_total']} 帖（{data['catalog_active']} 帖 24h 内有回复）",
        f"- AI 相关 {data['ai_total']} 帖，其中 {data['active_total']} 帖 24h 内有回帖",
        "",
    ]
    for it in data["threads"]:
        th = it["thread"]
        out.append(f"## {th['title']}")
        out.append(f"- 链接：{th['url']}")
        out.append(f"- 最后回复 {th['last_post']}，24h 回帖 {it['posts_total_24h']} 条")
        for p in it["posts"]:
            out.append(f"[{p['time']}] {p['text']}")
        out.append("")

    idle = data.get("idle") or []
    if idle:
        out.append("## 24h 内暂无新回复的 AI 帖（只有标题，无正文）")
        for t in idle:
            out.append(f"- {t['title']}（最后回复 {t.get('last_post') or '未知'}）")
        out.append("")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="/tmp/s1_posts.txt")
    ap.add_argument("--hours", type=int, default=24)
    args = ap.parse_args()

    text = build_text(args.hours)
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(f"\n写出 {p}：{len(text)} 字符 / {len(text.splitlines())} 行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
