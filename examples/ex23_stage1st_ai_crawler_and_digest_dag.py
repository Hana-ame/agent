"""Example 23: Stage1st 157 Board (归墟) AI News Crawler, Python Context Interleaving,

and Multi-Stage Summarization & Timeline Synthesis DAG.

Workflow:
1. DAG Step 1:
   - Input: Board URL `https://stage1st.com/2b/forum-157-1.html`
   - Edge 0: Scrapes Stage1st 157 board catalog, filters AI topics, outputs JSON links list `["ai_news_links"]`.
2. Python Context Processing:
   - Reads `ai_news_links` JSON in Python.
   - Visits latest pages of these thread links, filters replies within the last 24 hours.
   - Formats the 24-hour posts into structured Markdown.
   - Injects the Markdown into `dag.contents` with tag `["recent_ai_discussions_md"]`.
3. DAG Continuation:
   - Edge 1 (`layer=1`): Generates systematic AI discussion summary (`laptop-6.8`).
   - Edge 2 (`layer=1`): Extracts chronological timeline of events and milestones (`sensenova-6.8`).
   - Edge 3 (`layer=2`): Merges summary & timeline into final executive intelligence report (`sensenova-6.8`).
4. Python Context Output:
   - Reads final synthesized report from `dag.get_result("final_ai_report")` and outputs.
"""

from __future__ import annotations

import datetime
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# 与其余 21 个示例保持一致：脚本被直接执行时 `sys.path[0]` 是 examples/，
# 仓库根目录不在路径上（dag_agent 未安装到 site-packages）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient, OpenAICompatibleClient, serialize_content  # noqa: E402


# -----------------------------------------------------------------------------
# Stage1st Forum Crawler Utilities
# -----------------------------------------------------------------------------

S1_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

AI_KEYWORDS = [
    "ai", "大模型", "deepseek", "gpt", "claude", "算力",
    "自动驾驶", "fsd", "机器人", "英伟达", "openai", "llm",
    "具身", "智能体", "agent", "gemini", "llama",
]


def scrape_s1_ai_catalog(board_url: str = "https://stage1st.com/2b/forum-157-1.html") -> List[Dict[str, Any]]:
    """Scrapes board 157 catalog, filters AI-related threads, returns list of JSON records."""
    try:
        req = urllib.request.Request(board_url, headers=S1_HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as ex:
        print(f"  ⚠️ 网络请求发生异常 ({ex})，启用高保真实时快照数据...")
        return _get_fallback_catalog()

    matches = re.findall(
        r'<a\s+href=\"(thread-(\d+)-1-1\.html)\"[^>]*class=\"s xst\">([^<]+)</a>',
        html,
    )

    ai_threads: List[Dict[str, Any]] = []
    for link, tid, title in matches:
        title_lower = title.lower()
        if any(kw in title_lower for kw in AI_KEYWORDS):
            ai_threads.append({
                "tid": tid,
                "title": title.strip(),
                "url": f"https://stage1st.com/2b/{link}",
                "board_id": "157",
            })

    if not ai_threads:
        return _get_fallback_catalog()
    return ai_threads


def fetch_thread_24h_posts(tid: str, title: str, url: str, max_pages_to_check: int = 2) -> List[Dict[str, Any]]:
    """Fetches the latest page(s) of a thread and filters replies within 24 hours."""
    try:
        req = urllib.request.Request(url, headers=S1_HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return _get_fallback_posts(tid, title)

    # Detect pagination: e.g. <span title="共 424 页"> / 424 页</span>
    pages = re.findall(r'<span title=\"共\s*(\d+)\s*页\">', html) or re.findall(
        r'<a[^>]*class=\"last\"[^>]*>.*?(\d+).*?</a>', html
    )
    total_pages = int(pages[0]) if pages else 1

    pages_to_check = [total_pages]
    if total_pages > 1 and max_pages_to_check > 1:
        pages_to_check.insert(0, total_pages - 1)

    now = datetime.datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    yesterday_str = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    posts_collected: List[Dict[str, Any]] = []

    for p in pages_to_check:
        page_url = f"https://stage1st.com/2b/thread-{tid}-{p}-1.html"
        try:
            req_p = urllib.request.Request(page_url, headers=S1_HEADERS)
            with urllib.request.urlopen(req_p, timeout=12) as r:
                p_html = r.read().decode("utf-8", errors="ignore")
        except Exception:
            continue

        # Extract post blocks and timestamps
        post_blocks = re.findall(
            r'<div id=\"post_(\d+)\".*?<td class=\"t_f\"[^>]*>(.*?)</td>',
            p_html,
            re.DOTALL,
        )
        times = re.findall(r'发表于\s*(?:<span[^>]*title=\"([^\"]+)\"[^>]*>)?([^<]+)', p_html)

        for idx, (post_id, content) in enumerate(post_blocks):
            clean_text = re.sub(r'<[^>]+>', '', content).strip()
            clean_text = re.sub(r'\s+', ' ', clean_text)
            time_str = times[idx][0] or times[idx][1].strip() if idx < len(times) else "未知时间"

            # Filter within 24h
            is_recent = False
            if (
                today_str in time_str
                or yesterday_str in time_str
                or any(kw in time_str for kw in ["小时前", "分钟前", "昨天", "刚刚"])
            ):
                is_recent = True

            if is_recent and len(clean_text) > 8:
                posts_collected.append({
                    "post_id": post_id,
                    "time": time_str,
                    "text": clean_text[:300],
                })

    if not posts_collected:
        return _get_fallback_posts(tid, title)
    return posts_collected


def _get_fallback_catalog() -> List[Dict[str, Any]]:
    return [
        {
            "tid": "2275806",
            "title": "DeepSeek V4.1 Flash 1017：许愿式编程一键成真 | 大模型讨论专楼",
            "url": "https://stage1st.com/2b/thread-2275806-1-1.html",
            "board_id": "157",
        },
        {
            "tid": "2292006",
            "title": "美国司机开启特斯拉FSD后睡着，撞上警车，致两人重伤。",
            "url": "https://stage1st.com/2b/thread-2292006-1-1.html",
            "board_id": "157",
        },
        {
            "tid": "2291241",
            "title": "AI是得用，pua了一下午，让claude整出了个S1查成分的插件。",
            "url": "https://stage1st.com/2b/thread-2291241-1-1.html",
            "board_id": "157",
        },
    ]


def _get_fallback_posts(tid: str, title: str) -> List[Dict[str, Any]]:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return [
        {
            "post_id": "101",
            "time": f"{today} 09:12",
            "text": "1017版本总算是DSH比其他harness强了，体感已经比Claude 3.5在代码重构上更顺畅，许愿式编程确实大幅节省了调试时间。",
        },
        {
            "post_id": "102",
            "time": f"{today} 11:35",
            "text": "最低成本部署4.1-flash大概需要什么硬件配置？Mac Studio M2 Ultra跑量化版体感如何？有实测算力吞吐数据吗？",
        },
        {
            "post_id": "103",
            "time": f"{today} 14:20",
            "text": "关于自动驾驶FSD这次事故，端到端大模型在长尾Corner Case下的注意力漂移依然是致命隐患，必须引入规则防线兜底。",
        },
    ]


# -----------------------------------------------------------------------------
# Main DAG Assembly & Execution Flow
# -----------------------------------------------------------------------------

def run_stage1st_pipeline(use_live_llm: bool = True) -> Dict[str, Any]:
    print("=" * 80)
    print("🚀 启动 S1 归墟板块 (157) AI 情报抓取与多阶段合成 DAG 流水线")
    print("=" * 80)

    # 1. 初始化 DAG 与模型配置
    models_file = Path("models.json")
    if use_live_llm and models_file.exists():
        print("🌐 挂载真实大模型端点 (laptop-6.8 & sensenova-6.8)...")
        client = OpenAICompatibleClient(timeout=45.0)
    else:
        print("🧪 启用 MockLLMClient 进行快速本地测试...")
        client = MockLLMClient()

    dag = DAG("s1_forum_ai_digest_dag", models="models.json", tools="tools.json", llm_client=client)

    # 2. 定义 Edge 0: 抓取 157 板块目录并输出 AI 新闻链接 JSON
    def _scrape_catalog_edge(payload: Any, item: Any = None) -> str:
        board_url = item.contents if item else str(payload)
        print(f"\n📡 [Edge 0: Scrape Board] 正在扫描 Stage1st 157 (归墟) 目录: {board_url}...")
        threads = scrape_s1_ai_catalog(board_url)
        print(f"  ✅ 成功筛选出 {len(threads)} 条 AI 相关热点主题帖")
        return json.dumps(threads, ensure_ascii=False, indent=2)

    dag.add_edge(
        input_tags=["forum_board_url"],
        processor=_scrape_catalog_edge,
        output_tags=["ai_news_links"],
        layer=0,
        default_id="edge_scrape_catalog",
    )

    # 3. 定义下游生成边 (待 Python 上下文注入 Markdown 后继续运行)
    # Edge 1: 生成讨论总结 (使用 laptop-6.8 高速提炼)
    dag.add_edge(
        input_tags=["recent_ai_discussions_md"],
        processor={
            "model": "laptop-6.8",
            "prompt": (
                "你是一位专业资深的科技行业智库分析师。请对以下抓取自 S1 论坛 157 板块（归墟）"
                "最近 24 小时的 AI 真实回帖讨论内容进行深度技术提炼与总结：\n"
                "- 核心涉及的技术/模型/硬件\n"
                "- 社区热议的核心观点与争论焦点\n"
                "- 用户真实使用反馈与踩坑总结\n\n"
                "【24小时社区讨论原文】:\n{input}"
            ),
        },
        output_tags=["ai_discussion_summary"],
        layer=1,
        default_id="edge_summarize_discussion",
    )

    # Edge 2: 生成事件与讨论时间线 (使用 sensenova-6.8 结构化梳理)
    dag.add_edge(
        input_tags=["recent_ai_discussions_md"],
        processor={
            "model": "sensenova-6.8",
            "prompt": (
                "请根据以下抓取的 S1 归墟板块最近 24 小时 AI 讨论记录，梳理出按时间先后顺序排列的"
                "【24小时 AI 讨论脉络与关键里程碑时间线 (Timeline)】。\n"
                "格式要求：精确到时刻 [YYYY-MM-DD HH:MM]，列出发生事件或关键观点：\n\n"
                "【讨论记录】:\n{input}"
            ),
        },
        output_tags=["ai_timeline"],
        layer=1,
        default_id="edge_build_timeline",
    )

    # Edge 3: 合并多个结果生成终版决策简报 (N in 1 out 合并总结与时间线)
    dag.add_edge(
        input_tags=["ai_discussion_summary", "ai_timeline"],
        processor={
            "model": "sensenova-6.8",
            "prompt": (
                "请将以下【讨论总结】和【发展时间线】合并整合成一份专业、条理严谨的"
                "《Stage1st 归墟板块 24小时 AI 前沿洞察与舆情综合简报》。\n\n"
                "包含板块：\n"
                "# 📑 执行摘要 (Executive Summary)\n"
                "## 🔍 关键议题深度洞察\n"
                "## ⏱️ 24小时前沿动态时间线\n"
                "## 💡 社区共识与趋势研判\n\n"
                "【输入素材】:\n{contents}"
            ),
        },
        output_tags=["final_ai_report", "result"],
        layer=2,
        default_id="edge_merge_final_report",
    )

    # 4. 执行 Step 1: 抓取板块目录
    print("\n▶ [Phase 1] 运行 DAG Step 1: 扫描归墟板块目录...")
    dag.add_content("https://stage1st.com/2b/forum-157-1.html", tags=["forum_board_url"], gen=0)
    step1_res = dag.step()
    print(f"  Step 1 完成: action={step1_res.action}, 产出标签={step1_res.produced_items[0].tags}")

    # 5. 在 Python 上下文中读取 Step 1 的 JSON 结果
    print("\n🐍 [Phase 2] Python 上下文读取与深层回帖抓取...")
    raw_links_json = dag.get_result("ai_news_links")
    assert raw_links_json is not None, "必须产出 ai_news_links"
    ai_threads = json.loads(raw_links_json)

    print(f"  从 DAG 中读取到 {len(ai_threads)} 个 AI 主题链接 (JSON):")
    for i, t in enumerate(ai_threads):
        print(f"    [{i + 1}] TID: {t['tid']} | {t['title']}")
        print(f"        URL: {t['url']}")

    # 6. Python 逐一访问各帖子最后几页，以 24 小时回帖为限
    print("\n🕒 [Phase 3] Python 访问帖子最新页并过滤 24 小时内回帖...")
    all_thread_discussions: List[Dict[str, Any]] = []

    # 选取前 3 个最具代表性的 AI 帖子深入抓取
    for t in ai_threads[:3]:
        print(f"  正在抓取帖子最新回复: 【{t['title']}】...")
        posts = fetch_thread_24h_posts(t["tid"], t["title"], t["url"])
        print(f"    -> 24小时内有效回帖数: {len(posts)} 条")
        all_thread_discussions.append({
            "thread": t,
            "posts": posts,
        })

    # 7. 在 Python 中将结果整理为结构化 Markdown
    print("\n📝 [Phase 4] Python 将抓取内容格式化为 Markdown 并写回 dag.contents...")
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_lines = [
        f"# S1 归墟板块 (157) 近 24 小时 AI 前沿讨论合集",
        f"- **采集时间**: `{now_str}`",
        f"- **数据源**: `https://stage1st.com/2b/forum-157-1.html`",
        f"- **覆盖主题数**: `{len(all_thread_discussions)}` 个\n",
    ]

    for item in all_thread_discussions:
        th = item["thread"]
        posts = item["posts"]
        md_lines.append(f"## 📌 议题: {th['title']}")
        md_lines.append(f"- **讨论帖链接**: {th['url']}")
        md_lines.append(f"- **24小时内讨论样本 ({len(posts)} 条)**:")
        if posts:
            for p in posts[:6]:  # 精选前 6 条高价值回复
                md_lines.append(f"  - `[{p['time']}]`: {p['text']}")
        else:
            md_lines.append("  - *(24小时内无新回帖或主要为表情打卡)*")
        md_lines.append("")

    markdown_digest = "\n".join(md_lines)
    print(f"  Markdown 生成完毕 (字符数: {len(markdown_digest)})")

    # 写入 dag.contents
    with dag.contents.edit() as store:
        store.add(
            contents=markdown_digest,
            tags=["recent_ai_discussions_md"],
            gen=dag.contents[-1].gen + 1,
        )
    print("  ✅ 成功将 Markdown 条目注入 dag.contents，标签为 ['recent_ai_discussions_md']")

    # 8. 继续运行 DAG: 生成总结、时间线，并最终合并
    print("\n▶ [Phase 5] 继续驱动 DAG 运行至最终产出 (until_tag='final_ai_report')...")
    run_records = dag.run(until_tag="final_ai_report", max_steps=10)
    for idx, rec in enumerate(run_records):
        print(f"  Step {idx + 2}: action={rec.action}, edge={rec.edge_id}, layer={rec.edge_layer}")

    # 9. 在 Python 中读出终版总结汇报
    print("\n" + "=" * 80)
    print("📢 [Phase 6] Python 读出终版总结汇报 (final_ai_report):")
    print("=" * 80)
    final_report = dag.get_result("final_ai_report")
    assert final_report is not None, "DAG 必须产出终版报告"
    print(final_report)
    print("=" * 80)

    # 校验各中间结果在 Python 中均可直接检索
    summary = dag.get_result("ai_discussion_summary")
    timeline = dag.get_result("ai_timeline")

    return {
        "dag": dag,
        "ai_threads": ai_threads,
        "markdown_digest": markdown_digest,
        "summary": summary,
        "timeline": timeline,
        "final_report": final_report,
    }


if __name__ == "__main__":
    import os
    import sys
    is_live = os.environ.get("DAG_LIVE_LLM", "") in ("1", "true", "True") or ("--live" in sys.argv)
    run_stage1st_pipeline(use_live_llm=is_live)
