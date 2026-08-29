"""finance_ai_report: stage1st 财经版块 -> LLM 筛选 -> 逐帖抓回复并总结的 MapEdge 管线。

与 s1_ai_report_map 同构:
- FetchThreadsEdge 抓版块最新帖子列表(同 s1_ai_report_map 的 stage1st 解析,直连 trust_env=False);
- FilterEdge 由 LLM 选出财经/投资/经济/商战/国际时政相关帖(不限定数量);
- ProcessThreadsMap 对每个候选帖并发走 pipeline: FetchEdge(抓最近 hours 回复) -> SummarizeEdge(中文小结);
- 结果逐条发给 v_report,由 ReportVertex 累加写 report.md。

主题差异:筛选与总结 prompt 从「AI 相关」换成「财经/投资/时政」,summarize 小节改为
【事件】【观点】【影响】,事件按时间顺序排列(与 s1 一致,时间戳来自楼层)。

抓取层直接复用 s1_ai_report_map/s1_edges.py 的 stage1st 解析(版块列表 + 楼层),
已修过的坑一并继承:post_rate_div_ 空占位排除、中文「发表于」时间戳解析、多页回帖
按时间升序排序。
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone

import httpx
from bs4 import BeautifulSoup

from framework.edge import Edge, MapEdge

logger = logging.getLogger(__name__)

# stage1st 走直连(trust_env=False),不信任环境代理 —— 与 s1_ai_report_map 一致
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_BASE = "https://stage1st.com/2b/"


async def fetch_forum_threads(url: str) -> str:
    """抓版块首页,返回帖子列表 JSON 字符串(tid/title/url)。"""
    async with httpx.AsyncClient(
        headers=_HEADERS, timeout=30, trust_env=False, follow_redirects=True
    ) as c:
        r = await c.get(url)
        r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    seen = set()
    for href in soup.select("a.xst"):
        h = str(href.get("href", ""))
        if "thread-" not in h:
            continue
        tid = h.split("-")[1]
        if tid in seen:
            continue
        seen.add(tid)
        title = href.get_text(strip=True)
        if title:
            out.append({"tid": tid, "title": title, "url": _BASE + h})
    return json.dumps(out, ensure_ascii=False)


def _parse_posts_from_soup(soup):
    """从一页帖子 HTML 抠出楼层列表 (orig_idx, user, timestr, dt, content)。

    踩坑(已在 s1 侧修过,这里继承):id="post_rate_div_<pid>" 是空的评分占位
    div,不能当楼层;真实楼层 id 匹配 ^post_[0-9]+$。时间戳是中文「发表于 …」,
    span[title] 常为空,需从 em#authorposton 文本里 re.search 解析。
    """
    posts = [
        d for d in soup.select('div[id^="post_"]')
        if re.match(r"^post_\d+$", d.get("id", ""))
    ]
    result = []
    for i, block in enumerate(posts):
        user = "Unknown"
        user_node = block.select_one(".authi a.xw1")
        if user_node:
            user = user_node.get_text(strip=True)

        time_str = ""
        dt = None
        em_node = block.select_one('em[id^="authorposton"]')
        if em_node:
            span = em_node.select_one("span[title]")
            if span and span.get("title"):
                time_str = span.get("title")
            else:
                time_str = em_node.get_text(strip=True)
            time_str = re.sub(r"^(发表于|Post on|Posted at)[\s:：]*", "", time_str).strip()
            try:
                m = re.search(
                    r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})[ T](\d{1,2}):(\d{2})", time_str
                )
                if m:
                    y, mo, d, h, mi = map(int, m.groups())
                    dt = datetime(y, mo, d, h, mi, tzinfo=timezone(timedelta(hours=8)))
            except Exception:
                pass

        msg = block.select_one('td.t_f, [id^="postmessage_"]')
        content = "(empty)"
        if msg:
            content = msg.get_text("\n", strip=True)
            content = re.sub(r"\n{3,}", "\n\n", content)

        result.append((i, user, time_str, dt, content))
    return result


def _extract_posts_from_soup(soup, url: str, hours: int) -> str:
    """单页帖子 → 最近 hours 小时楼层的 markdown(用于无分页/单页场景)。"""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    title = url
    for sel in ("#thread_subject", "h1.ts2", "h1.title"):
        node = soup.select_one(sel)
        if node and node.get_text(strip=True):
            title = node.get_text(strip=True)
            break

    recent = []
    for i, user, timestr, dt, content in _parse_posts_from_soup(soup):
        if dt and dt >= cutoff:
            recent.append((i, user, timestr, content))

    lines = [f"# {title}", "", f"> Link: <{url}>", f"> Result: **{len(recent)}** posts", "", "---", ""]
    for orig_idx, user, timestr, content in recent:
        lines.extend([f"### #{orig_idx + 1} **{user}** · {timestr}", "", content, "", "---", ""])
    return "\n".join(lines).rstrip() + "\n"


async def fetch_thread_replies_md(url: str, hours: int = 24, timeout: float = 30) -> str:
    """抓一个 stage1st 帖子的所有页,过滤最近 *hours* 小时楼层,输出 markdown。

    多页按「倒序翻页、到无新帖那页就停」;楼层按时间升序排序后输出,方便
    总结 LLM 按事件时间线写。
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    async def _get(u: str):
        async with httpx.AsyncClient(
            headers=_HEADERS, timeout=timeout, trust_env=False, follow_redirects=True
        ) as c:
            return await c.get(u)

    m = re.search(r"(thread-\d+-)(\d+)(-\d+\.html)", url)
    if not m:
        r = await _get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        return _extract_posts_from_soup(soup, url, hours)

    base_prefix = m.group(1)
    base_suffix = m.group(3)

    r1 = await _get(url)
    soup1 = BeautifulSoup(r1.text, "html.parser")

    title = url
    for sel in ("#thread_subject", "h1.ts2", "h1.title"):
        node = soup1.select_one(sel)
        if node and node.get_text(strip=True):
            title = node.get_text(strip=True)
            break

    total_pages = 1
    for a in soup1.select(".pg a[href]"):
        href = a.get("href", "")
        pm = re.search(r"-(\d+)-1\.html", href)
        if pm:
            pnum = int(pm.group(1))
            if pnum > total_pages:
                total_pages = pnum

    all_posts = []
    for page in range(total_pages, 0, -1):
        page_url = f"{_BASE}{base_prefix}{page}{base_suffix}"
        if page == 1:
            soup = soup1
        else:
            r = await _get(page_url)
            soup = BeautifulSoup(r.text, "html.parser")

        page_posts = _parse_posts_from_soup(soup)
        page_has_recent = False
        for orig_idx, user, timestr, dt, content in page_posts:
            if dt and dt >= cutoff:
                all_posts.append((dt, orig_idx, user, timestr, content))
                page_has_recent = True

        if not page_has_recent:
            break

    # 时间升序,让总结 LLM 能按时间线写【事件】小节
    all_posts.sort(key=lambda x: x[0])

    lines = [
        f"# {title}", "",
        f"> Link: <{url}>",
        f"> Range: Last **{hours} hours**",
        f"> Result: **{len(all_posts)}** replies", "", "---", "",
    ]

    if not all_posts:
        lines.append(f"_No replies in the last {hours} hours._")
        return "\n".join(lines)

    for _dt, orig_idx, user, timestr, content in all_posts:
        lines.extend([f"### #{orig_idx + 1} **{user}** · {timestr}", "", content, "", "---", ""])

    return "\n".join(lines).rstrip() + "\n"


class FetchThreadsEdge(Edge):
    """抓版块首页帖子列表(JSON 字符串),post_process 解析成 list[dict]。"""

    async def pre_process(self, data, settings):
        return await fetch_forum_threads(str(data))

    def post_process(self, data, settings):
        try:
            m = re.search(r"\[.*\]", data, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return json.loads(data)
        except Exception:
            return []


class FilterEdge(Edge):
    """LLM 筛选结果的 JSON 解析(过滤逻辑在 config settings.prompt 里)。"""

    def post_process(self, data, settings):
        try:
            m = re.search(r"\[.*\]", data, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return json.loads(data)
        except Exception:
            return []


class FetchEdge(Edge):
    """取单个候选帖的最近回复(构造 {title,url,content} 给 SummarizeEdge)。"""

    def condition(self, data, settings):
        return isinstance(data, dict) and "url" in data

    async def pre_process(self, data, settings):
        hours = int(settings.get("hours", 24))
        timeout = float(settings.get("timeout", 30))
        md = await fetch_thread_replies_md(data["url"], hours=hours, timeout=timeout)
        return {"title": data.get("title", ""), "url": data.get("url", ""), "content": md}


class SummarizeEdge(Edge):
    """LLM 把单帖回复 markdown 提炼成中文小结;结构化标题/链接来自抓取数据。"""

    def pre_process(self, data, settings):
        if isinstance(data, dict):
            # 记住抓取来的原始标题/链接,报告不依赖 LLM 复述标题(省 token)
            self._title = data.get("title", "Unknown")
            self._url = data.get("url", "")
            content = data.get("content", "")
            return f"Thread Title: {self._title}\nLink: {self._url}\n\n{content}"
        return str(data)

    def post_process(self, data, settings):
        # 把 LLM 摘要和抓取来的标题/链接拼成结构化结果,供 ReportVertex 渲染
        summary = str(data)
        if isinstance(data, dict) and data.get("summary"):
            summary = data["summary"]
        return {
            "title": getattr(self, "_title", "Unknown"),
            "url": getattr(self, "_url", ""),
            "summary": summary,
        }


class ProcessThreadsMap(MapEdge):
    """MapEdge:对筛选出的帖子列表并发走 fetch+summarize pipeline(config 里定义)。"""

    pass