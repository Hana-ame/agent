import json
import httpx
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta, timezone
from framework.edge import Edge, MapEdge


async def fetch_forum_threads(url: str):
    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient(headers=headers, timeout=30, trust_env=False, follow_redirects=True) as c:
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
            out.append({"tid": tid, "title": title, "url": "https://stage1st.com/2b/" + h})
    return json.dumps(out, ensure_ascii=False)


async def fetch_thread_replies_md(url: str, hours: int = 24) -> str:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    headers = {"User-Agent": "Mozilla/5.0"}

    m = re.search(r"(thread-\d+-)(\d+)(-\d+\.html)", url)
    if not m:
        async with httpx.AsyncClient(headers=headers, timeout=30, trust_env=False, follow_redirects=True) as c:
            r = await c.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        return _extract_posts_from_soup(soup, url, hours)

    base_prefix = m.group(1)
    base_suffix = m.group(3)

    async with httpx.AsyncClient(headers=headers, timeout=30, trust_env=False, follow_redirects=True) as c:
        r1 = await c.get(url)
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
            page_url = f"https://stage1st.com/2b/{base_prefix}{page}{base_suffix}"
            if page == 1:
                soup = soup1
            else:
                r = await c.get(page_url)
                soup = BeautifulSoup(r.text, "html.parser")

            page_posts = _parse_posts_from_soup(soup)
            page_has_recent = False
            for orig_idx, user, timestr, dt, content in page_posts:
                if dt and dt >= cutoff:
                    all_posts.insert(0, (orig_idx, user, timestr, content))
                    page_has_recent = True

            if not page_has_recent:
                break

    lines = [f"# {title}", "", f"> Link: <{url}>",
             f"> Range: Last **{hours} hours**",
             f"> Result: **{len(all_posts)}** replies", "", "---", ""]

    if not all_posts:
        lines.append(f"_No replies in the last {hours} hours._")
        return "\n".join(lines)

    for orig_idx, user, timestr, content in all_posts:
        lines.extend([f"### #{orig_idx + 1} **{user}** · {timestr}", "", content, "", "---", ""])

    return "\n".join(lines).rstrip() + "\n"


def _parse_posts_from_soup(soup):
    posts = soup.select('div[id^="post_"]')
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
            if span:
                time_str = span.get("title")
            else:
                time_str = em_node.get_text(strip=True).replace("Post on", "").replace("Posted at", "").strip()
            try:
                if re.match(r"\d{4}-\d{1,2}-\d{1,2}", time_str):
                    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                    dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
            except:
                pass

        msg = block.select_one('td.t_f, [id^="postmessage_"]')
        content = "(empty)"
        if msg:
            content = msg.get_text("\n", strip=True)
            content = re.sub(r"\n{3,}", "\n\n", content)

        result.append((i, user, time_str, dt, content))
    return result


def _extract_posts_from_soup(soup, url, hours):
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


class FetchThreadsEdge(Edge):
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
    def post_process(self, data, settings):
        try:
            m = re.search(r"\[.*\]", data, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return json.loads(data)
        except Exception:
            return []


class SelectEdge(Edge):
    def condition(self, data, settings):
        index = int(settings.get("index", 0))
        return isinstance(data, list) and index < len(data)

    def pre_process(self, data, settings):
        index = int(settings.get("index", 0))
        return data[index]


class FetchEdge(Edge):
    def condition(self, data, settings):
        return isinstance(data, dict) and "url" in data

    async def pre_process(self, data, settings):
        hours = int(settings.get("hours", 24))
        md = await fetch_thread_replies_md(data["url"], hours=hours)
        return {"title": data.get("title", ""), "url": data.get("url", ""), "content": md}


class SummarizeEdge(Edge):
    def pre_process(self, data, settings):
        if isinstance(data, dict):
            title = data.get("title", "Unknown")
            url = data.get("url", "")
            content = data.get("content", "")
            return f"Thread Title: {title}\nLink: {url}\n\n{content}"
        return str(data)


class ProcessThreadsMap(MapEdge):
    """MapEdge — processes the filtered thread list concurrently through a
    fetch-replies + LLM-summarize pipeline (defined in config.json)."""
    pass
