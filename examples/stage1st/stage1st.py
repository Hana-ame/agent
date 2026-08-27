"""stage1st 资讯收集任务 —— 代码扩展的 Edge 实现。

把 chatto-bot 里的「逛 stage1st 拿 AI 资讯」固定脚本逻辑，
用本框架的「继承 Edge」机制写成代码扩展。

流程（纯脚本边 与 agent 边分离）：
    trigger → [IndexEdge]       抓整个版块 index(全部帖子) → index 顶点 (posts, [])
    index   → [SelectAIEdge]    agent 边：从全部帖子中筛选 AI 相关 → 输出 JSON
                                → picked 顶点 (picked, []) 接收筛选结果
    picked  → [ThreadFetchEdge] 按边 ID 分发第 1/2/3 个 → 从最后页往回爬 → docs (post_i)
    docs    → [ThreadAgentEdge] agent 边：读文本文档 → AI 总结 → summary
    summary → [DigestEdge]      agent 边：读全部 → AI 汇总播报 → final

爬取层：优先用 chatto-bot 的真实 plugins.s1parse（需 httpx/bs4，
        建议用 ../chatto-bot/.venv/bin/python 运行）；不可用时回退到内置 mock。
"""

import asyncio  # noqa: F401
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Any

from framework.edge import Edge
from framework.signal import AbortSignal, is_abort, abort_reason

logger = logging.getLogger("vertex_edge_agent.stage1st")

# chatto-bot 位于同级的 sibling 目录 /mnt/d/WorkPlace/chatto-bot
CHATTO_BOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "chatto-bot")
)
FORUM_URL = "https://stage1st.com/2b/forum-157-1.html"  # AI / 科技版块
THREAD_URL_TMPL = "https://stage1st.com/2b/thread-{tid}-{n}-1.html"

# ── 尝试加载真实爬取层(plugins.s1parse)，失败则用内置 mock ──
HAVE_REAL_PARSE = False
try:
    import httpx  # noqa: F401   httpx/bs4 在 chatto-bot 的 .venv 里
    if CHATTO_BOT_DIR not in sys.path:
        sys.path.insert(0, CHATTO_BOT_DIR)
    import plugins.s1parse as s1parse  # noqa: F401
    HAVE_REAL_PARSE = True
except Exception as _e:  # pragma: no cover - 依赖缺失时静默回退
    s1parse = None


# ----------------------------------------------------------------------
# 内置 mock(真实环境替换为 plugins.s1parse；逻辑同 chatto 的 ai_tour_graph)
# ----------------------------------------------------------------------
async def mock_index(forum_url: str) -> list[dict]:
    now = datetime.now()

    def t(hours: float) -> str:
        return (now - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M")

    return [
        {"tid": "t1", "title": "【AI】某开源模型发布,benchmark 屠榜",
         "url": f"{forum_url}#t1", "last_reply": t(0.1)},
        {"tid": "t2", "title": "新 Agent 框架讨论串",
         "url": f"{forum_url}#t2", "last_reply": t(2)},
        {"tid": "t3", "title": "AI 算力卡价格又涨了",
         "url": f"{forum_url}#t3", "last_reply": t(5)},
        {"tid": "t4", "title": "老帖:5 天前没人回复",
         "url": f"{forum_url}#t4", "last_reply": t(5 * 24)},
        {"tid": "t5", "title": "今天午饭吃什么(无关)",
         "url": f"{forum_url}#t5", "last_reply": t(1)},
    ]


async def mock_thread(url: str) -> list[dict]:
    return [
        {"uid": "1001", "username": "a", "time": "2026-08-24 12:00",
         "content": f"这是 {url.split('#')[-1]} 的第一层回复,讨论细节…"},
        {"uid": "1002", "username": "b", "time": "2026-08-24 12:30",
         "content": "补充:实测跑分比上代高 40%"},
        {"uid": "1003", "username": "c", "time": "2026-08-24 13:00",
         "content": "价格/生态方面怎么看?"},
    ]


# ----------------------------------------------------------------------
# 时间解析 / 公共工具
# ----------------------------------------------------------------------
def parse_last_reply(s: str | None) -> datetime | None:
    """解析 last_reply 文本，如 '2026-8-27 01:17' -> datetime。"""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def pick_recent(threads: list[dict], recent_hours: int = 24, max_threads: int = 3) -> list[dict]:
    """兜底：按 last_reply 挑最近 recent_hours 内的帖子(agent 筛选失败时用)。"""
    now = datetime.now()
    picked = [t for t in threads if (dt := parse_last_reply(t.get("last_reply")))
              and now - dt <= timedelta(hours=recent_hours)]
    picked.sort(key=lambda t: parse_last_reply(t.get("last_reply")), reverse=True)
    return picked[:max_threads]


def parse_json_list(raw: str | None) -> list:
    """从 agent 输出中解析出 JSON 数组(容忍 markdown 代码块 / 前后杂文)。"""
    if not raw:
        return []
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    s, e = raw.find("["), raw.rfind("]")
    if s == -1 or e == -1 or e <= s:
        return []
    try:
        return json.loads(raw[s:e + 1])
    except Exception:
        return []


def resolve_picked(items: list, threads: list[dict]) -> list[dict]:
    """把 agent 返回的条目对齐到原始帖子 dict(按 url/tid/title 匹配)。"""
    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        for t in threads:
            if (it.get("url") and it["url"] == t.get("url")) or \
               (it.get("tid") and str(it["tid"]) == str(t.get("tid"))) or \
               (it.get("title") and it["title"] == t.get("title")):
                out.append(t)
                break
        else:
            out.append(it)  # 匹配不到也保留(含 url 即可继续爬)
    return out


# ----------------------------------------------------------------------
# 帖子正文抓取：从最后一页往回爬，收齐 recent_hours 内所有回复
# ----------------------------------------------------------------------
async def fetch_recent_posts(url: str, recent_hours: int = 24) -> list[dict]:
    """从帖子最后一页往回爬，收集 recent_hours 小时内所有回复楼层。

    若 recent_hours 内没有新回复，降级抓最后一页(最新楼层)兜底，避免流程中断。
    """
    if not HAVE_REAL_PARSE:
        return await mock_thread(url)

    sp = s1parse
    threshold = datetime.now() - timedelta(hours=recent_hours)
    collected: list[dict] = []
    async with sp._client() as c:
        # 1) 首页：拿总页数、tid 与首页楼层
        r = await c.get(url)
        r.raise_for_status()
        posts, total = sp._parse_thread_html(r.text, 1)
        tid = sp._cur_tid(r.text)
        for p in posts:
            dt = sp._parse_stage1st_time(p.get("time"))
            if dt is not None and dt >= threshold:
                collected.append(p)
        # 2) 从最后一页往回爬
        if tid:
            for n in range(total, 1, -1):
                rr = await c.get(THREAD_URL_TMPL.format(tid=tid, n=n))
                if rr.status_code != 200:
                    break
                pg, _ = sp._parse_thread_html(rr.text, n)
                dts = [sp._parse_stage1st_time(p.get("time")) for p in pg]
                if dts and all(d is not None and d < threshold for d in dts):
                    break  # 整页都早于阈值，再往前只会更旧
                for p, dt in zip(pg, dts):
                    if dt is not None and dt >= threshold:
                        collected.append(p)
        # 3) 降级：24h 内无新回复，抓最后一页(最新楼层)兜底，保证有文档产出
        if not collected and tid and total >= 1:
            logger.debug("[fetch_recent_posts] 近 %dh 无新回复，降级抓最后一页(p%d)", recent_hours, total)
            rr = await c.get(THREAD_URL_TMPL.format(tid=tid, n=total))
            if rr.status_code == 200:
                pg, _ = sp._parse_thread_html(rr.text, total)
                collected.extend(pg)
    return collected


def format_thread_doc(thread: dict, posts: list[dict]) -> str:
    """把帖子与回复楼层整理成一份纯文本文档。"""
    lines = [
        f"帖子标题: {thread.get('title', '')}",
        f"链接: {thread.get('url', '')}",
        "",
    ]
    for p in posts:
        user = p.get("username") or p.get("user") or "?"
        lines.append(f"[{user} @ {p.get('time', '')}]")
        lines.append(p.get("content", ""))
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 自定义 Edge 实现
# ----------------------------------------------------------------------
class IndexEdge(Edge):
    """抓取整个版块 index(不筛选)，把全部帖子写入 index 顶点 (posts, [])。"""

    def __init__(self, edge_id, source_id, destination_id,
                 forum_url: str = FORUM_URL, **kwargs):
        super().__init__(edge_id, source_id, destination_id, **kwargs)
        self.forum_url = forum_url

    async def _fetch_index(self) -> list[dict]:
        if HAVE_REAL_PARSE:
            return await s1parse.index_to_json(self.forum_url)
        return await mock_index(self.forum_url)

    async def execute(self, source_vertex, dest_vertex, pi_agent) -> Any:
        # 1) 读 trigger 的 tick(只做同步起点)
        await source_vertex.get("tick", [])
        # 2) 抓整个 index(全部帖子)
        threads = await self._fetch_index()
        if not threads:
            raise RuntimeError("[IndexEdge] 版块没有帖子")
        # 3) 写入 index 顶点 (posts, [])
        await dest_vertex.set(threads, "posts", [])
        self.completed, self.result = True, [t["title"] for t in threads]
        logger.debug("[IndexEdge] 抓到 %d 个帖子", len(threads))
        return self.result


class SelectAIEdge(Edge):
    """agent 边：从全部帖子中筛选 AI 相关，输出 JSON 数组 → picked 顶点 (picked, [])。"""

    def __init__(self, edge_id, source_id, destination_id,
                 max_threads: int = 3, **kwargs):
        super().__init__(edge_id, source_id, destination_id, **kwargs)
        self.max_threads = max_threads

    async def execute(self, source_vertex, dest_vertex, pi_agent) -> Any:
        # 1) 读 index 顶点全部帖子
        threads = await source_vertex.get("posts", [])
        if not threads:
            raise RuntimeError("[SelectAIEdge] index 顶点没有帖子")
        # 2) 拼成清单交给 agent 筛选
        listing = "\n".join(
            f"{i + 1}. {t.get('title', '')} | {t.get('url', '')} | last_reply:{t.get('last_reply', '')}"
            for i, t in enumerate(threads)
        )
        text = f"以下是 stage1st 版块帖子清单:\n{listing}"
        raw = await pi_agent.process(text, self.prompt, self.model, self.settings)

        # 3) 解析 agent 输出的 JSON，对齐到原始帖子
        items = parse_json_list(raw)
        picked = resolve_picked(items, threads) if items else []
        # 容错：解析失败/为空则按最近回复兜底挑前 max_threads 个
        if not picked:
            logger.warning("[SelectAIEdge] agent 未返回可用 JSON，按最近回复兜底挑选")
            picked = pick_recent(threads, max_threads=self.max_threads)
        picked = picked[:self.max_threads]

        # 4) 写入 picked 顶点 (picked, []) —— 接收筛选结果的顶点
        #    若筛选数量不足 max_threads(期望的 fan-out 个数)，生成 Abort 信号 + reason
        if len(picked) < self.max_threads:
            reason = (f"agent 仅筛选到 {len(picked)} 个 AI 相关帖，"
                      f"少于所需 fan-out 个数 {self.max_threads}，流程中止")
            logger.warning("[SelectAIEdge] %s", reason)
            return await self.forward_abort(dest_vertex, reason, "picked", [])
        await dest_vertex.set(picked, "picked", [])
        self.completed, self.result = True, [t.get("title") for t in picked]
        logger.debug("[SelectAIEdge] 筛选出 %d 个 AI 相关帖", len(picked))
        return self.result


class ThreadFetchEdge(Edge):
    """纯脚本边：按边 ID 分发第 post_index 个 → 从最后页往回爬 → 输出文本文档。

    读 picked 顶点 (picked, [])；把 24h 内所有回复整理成文本文档，
    写入 docs 顶点 (data_id, tags) —— 捕捉顶点承接。
    """

    def __init__(self, edge_id, source_id, destination_id,
                 post_index: int = 0, recent_hours: int = 24, **kwargs):
        super().__init__(edge_id, source_id, destination_id, **kwargs)
        self.post_index = post_index
        self.recent_hours = recent_hours

    async def execute(self, source_vertex, dest_vertex, pi_agent) -> Any:
        # 0) 若上游已是 Abort 信号，直接透传，不爬取
        val = await source_vertex.get("picked", [])
        if is_abort(val):
            return await self.forward_abort(dest_vertex, val.reason, self.data_id, self.tags)
        picked = val
        # 1) 从 picked 顶点取筛选结果，按本边序号取第 post_index 个(分发)
        if not picked or self.post_index >= len(picked):
            raise RuntimeError(f"[ThreadFetchEdge:{self.id}] picked[{self.post_index}] 不存在")
        thread = picked[self.post_index]
        # 2) 纯脚本爬取：从最后一页往回，收齐 recent_hours 内所有回复
        posts = await fetch_recent_posts(thread["url"], self.recent_hours)
        if not posts:
            raise RuntimeError(f"[ThreadFetchEdge:{self.id}] 近 {self.recent_hours}h 内没有回复")
        # 3) 整理成文本文档，写入 docs 顶点(捕捉)
        doc = format_thread_doc(thread, posts)
        await dest_vertex.set(doc, self.data_id, self.tags)
        self.completed, self.result = True, doc
        logger.debug("[ThreadFetchEdge:%s] 分发帖子[%d] 爬取 %d 层 → docs (%s)",
                    self.id, self.post_index, len(posts), self.data_id)
        return doc


class ThreadAgentEdge(Edge):
    """agent 边：读 docs 顶点 (data_id, tags) 的文本文档 → AI 总结 → 写入 summary 顶点。"""

    async def execute(self, source_vertex, dest_vertex, pi_agent) -> Any:
        # 0) 若上游已是 Abort 信号，直接透传，不调 agent
        val = await source_vertex.get(self.data_id, self.tags)
        if is_abort(val):
            return await self.forward_abort(dest_vertex, val.reason, self.data_id, self.tags)
        doc = val
        if doc is None:
            raise RuntimeError(f"[ThreadAgentEdge:{self.id}] docs ({self.data_id},{self.tags}) 无数据")
        result = await pi_agent.process(doc, self.prompt, self.model, self.settings)
        await dest_vertex.set(result, self.data_id, self.tags)
        self.completed, self.result = True, result
        logger.debug("[ThreadAgentEdge:%s] 已总结文档 (%s)", self.id, self.data_id)
        return result


class DigestEdge(Edge):
    """agent 边：读 summary 顶点全部数据 → AI 汇总成播报 → 写入 final 顶点 (final, [])。"""

    async def execute(self, source_vertex, dest_vertex, pi_agent) -> Any:
        all_data = await source_vertex.get_all_data()
        if not all_data:
            raise RuntimeError("[DigestEdge] summary 顶点没有数据")
        # 0) 若上游任一分槽是 Abort 信号，汇总 reason 后透传给 final
        aborts = {abort_reason(v) for v in all_data.values() if is_abort(v)}
        if aborts:
            reason = "；".join(sorted(aborts)) or "上游 Abort"
            return await self.forward_abort(dest_vertex, reason, "final", [])
        parts = []
        for key in sorted(all_data.keys()):
            parts.append(f"[{key[0]}:{','.join(key[1])}]\n{all_data[key]}")
        text = "以下是多个 stage1st 帖子的独立总结:\n\n" + "\n\n".join(parts)
        result = await pi_agent.process(text, self.prompt, self.model, self.settings)
        await dest_vertex.set(result, "final", [])
        self.completed, self.result = True, result
        return result
