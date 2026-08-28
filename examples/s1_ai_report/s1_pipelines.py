import json
import httpx
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta

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
    return json.dumps(out[:8], ensure_ascii=False)  # Top 8 threads for 8 edges

async def fetch_thread_replies_md(url: str, hours: int = 24) -> str:
    now = datetime.now()
    cutoff = now - timedelta(hours=hours)
    
    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient(headers=headers, timeout=30, trust_env=False, follow_redirects=True) as c:
        r = await c.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    
    title = url
    for sel in ("#thread_subject", "h1.ts2", "h1.title"):
        node = soup.select_one(sel)
        if node and node.get_text(strip=True):
            title = node.get_text(strip=True)
            break
            
    recent = []
    posts = soup.select('div[id^="post_"]')
    
    for i, block in enumerate(posts):
        user = "Unknown"
        user_node = block.select_one('.authi a.xw1')
        if user_node:
            user = user_node.get_text(strip=True)
            
        time_str = ""
        dt = None
        em_node = block.select_one('em[id^="authorposton"]')
        if em_node:
            span = em_node.select_one('span[title]')
            if span:
                time_str = span.get('title')
            else:
                time_str = em_node.get_text(strip=True).replace('发表于', '').strip()
            
            try:
                # "2024-8-20 10:00"
                if re.match(r'\d{4}-\d{1,2}-\d{1,2}', time_str):
                    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
            except:
                pass
                
        # If time parsing fails, fallback to current time just to include it (or skip it)
        if dt is None:
            continue
            
        if dt >= cutoff:
            msg = block.select_one("td.t_f, [id^='postmessage_']")
            content = "(空)"
            if msg:
                content = msg.get_text("\n", strip=True)
                content = re.sub(r"\n{3,}", "\n\n", content)
            
            recent.append((i, user, time_str, content))

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"> 链接: <{url}>")
    lines.append(f"> 范围: 最近 **{hours} 小时** 的回复")
    lines.append(f"> 结果: **{len(recent)}** / {len(posts)} 楼")
    lines.append("")
    lines.append("---")
    lines.append("")

    if not recent:
        lines.append(f"_最近 {hours} 小时暂无新回复。_")
        return "\n".join(lines)

    for seq, (orig_idx, user, timestr, content) in enumerate(recent, 1):
        lines.append(f"### #{orig_idx + 1} **{user}** · {timestr}")
        lines.append("")
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"

class FilterPipeline:
    @staticmethod
    async def pre_process(data, settings):
        return await fetch_forum_threads(str(data))

    @staticmethod
    def post_process(data, settings):
        try:
            m = re.search(r'\[.*\]', data, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return json.loads(data)
        except Exception:
            return []

class ThreadPipeline:
    @staticmethod
    def condition(data, settings):
        index = int(settings.get("index", 0))
        if not isinstance(data, list) or index >= len(data):
            return False
        return True

    @staticmethod
    async def pre_process(data, settings):
        index = int(settings.get("index", 0))
        thread = data[index]
        hours = int(settings.get("hours", 24))
        md = await fetch_thread_replies_md(thread["url"], hours=hours)
        return md
