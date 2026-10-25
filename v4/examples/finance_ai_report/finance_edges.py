"""finance_ai_report: Forum financial section -> LLM filter -> thread fetch and summary MapEdge pipeline.

Isomorphic to s1_ai_report_map:
- FetchThreadsEdge fetches latest thread list from the forum section (trust_env=False);
- FilterEdge uses LLM to select financial/investment/economic/geopolitical threads;
- ProcessThreadsMap runs concurrent pipeline for each candidate: FetchEdge -> SummarizeEdge;
- Results are delivered to v_report, where ReportVertex accumulates and writes report.md.
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

# Direct connection (trust_env=False), ignoring environment proxy
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_BASE = "https://stage1st.com/2b/"


async def fetch_forum_threads(url: str) -> str:
    """Fetch section index page and return thread list as JSON string (tid/title/url)."""
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
    """Extract post list from thread HTML page (orig_idx, user, timestr, dt, content).

    id="post_rate_div_<pid>" is an empty rating div placeholder and must not be
    counted as a post; real post IDs match ^post_[0-9]+$.
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
            time_str = re.sub(r"^(Post on|Posted at)[\s:\uFF1A]*", "", time_str).strip()
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
    """Extract posts within recent hours for single-page threads."""
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
    """Fetch all pages of a thread, filter posts from recent hours, and output markdown."""
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

    # Chronological order
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
    """Fetch forum thread list JSON string and parse into list[dict]."""

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
    """Parse LLM filtering JSON response."""

    def post_process(self, data, settings):
        try:
            m = re.search(r"\[.*\]", data, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return json.loads(data)
        except Exception:
            return []


class FetchEdge(Edge):
    """Fetch recent replies for a candidate thread ({title, url, content})."""

    def condition(self, data, settings):
        return isinstance(data, dict) and "url" in data

    async def pre_process(self, data, settings):
        hours = int(settings.get("hours", 24))
        timeout = float(settings.get("timeout", 30))
        md = await fetch_thread_replies_md(data["url"], hours=hours, timeout=timeout)
        return {"title": data.get("title", ""), "url": data.get("url", ""), "content": md}


class SummarizeEdge(Edge):
    """Summarize thread replies with structured title/url preserved from fetched data."""

    def pre_process(self, data, settings):
        if isinstance(data, dict):
            self._title = data.get("title", "Unknown")
            self._url = data.get("url", "")
            content = data.get("content", "")
            return f"Thread Title: {self._title}\nLink: {self._url}\n\n{content}"
        return str(data)

    def post_process(self, data, settings):
        summary = str(data)
        if isinstance(data, dict) and data.get("summary"):
            summary = data["summary"]
        return {
            "title": getattr(self, "_title", "Unknown"),
            "url": getattr(self, "_url", ""),
            "summary": summary,
        }


class ProcessThreadsMap(MapEdge):
    """MapEdge: processes filtered thread list through fetch + summarize pipeline."""
    pass
