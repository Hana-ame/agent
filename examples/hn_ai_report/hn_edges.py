import json
import httpx
import re
import asyncio
import logging
from html import unescape
from datetime import datetime, timezone
from framework.edge import Edge, MapEdge

logger = logging.getLogger(__name__)

async def fetch_hn_top_stories(limit: int = 30, timeout: float = 30):
    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient(headers=headers, timeout=timeout) as c:
        r = await c.get("https://hacker-news.firebaseio.com/v0/topstories.json")
        r.raise_for_status()
        story_ids = r.json()[:limit]

        async def fetch_item(item_id):
            try:
                res = await c.get(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json")
                res.raise_for_status()
                return res.json()
            except Exception as e:
                logger.warning(f"Failed to fetch story {item_id}: {e}")
                return None

        items = await asyncio.gather(*(fetch_item(i) for i in story_ids))
    
    out = []
    for item in items:
        if not item or item.get("type") != "story":
            continue
        out.append({
            "id": item["id"],
            "title": item.get("title", ""),
            "url": item.get("url", f"https://news.ycombinator.com/item?id={item['id']}"),
            "score": item.get("score", 0)
        })
    return json.dumps(out, ensure_ascii=False)


async def fetch_hn_comments_md(story_id: int, max_comments: int = 15, timeout: float = 30) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient(headers=headers, timeout=timeout) as c:
        try:
            r = await c.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
            r.raise_for_status()
            story = r.json()
        except Exception as e:
            logger.warning(f"Failed to fetch story {story_id}: {e}")
            return "_Story not found._"
            
        if not story:
            return "_Story not found._"
        
        kids = story.get("kids", [])[:max_comments]
        if not kids:
            return "_No comments._"

        async def fetch_comment(item_id):
            try:
                res = await c.get(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json")
                res.raise_for_status()
                return res.json()
            except Exception as e:
                logger.warning(f"Failed to fetch comment {item_id}: {e}")
                return None

        comments = await asyncio.gather(*(fetch_comment(i) for i in kids))

    lines = [f"# {story.get('title', 'Unknown')}", "", f"> URL: <{story.get('url', '')}>", f"> HN Link: <https://news.ycombinator.com/item?id={story_id}>", "", "---", ""]
    
    for c in comments:
        if not c or c.get("type") != "comment" or c.get("deleted") or c.get("dead"):
            continue
        user = c.get("by", "Unknown")
        text = c.get("text", "")
        # Robust HTML cleaning using html.unescape
        text = text.replace("<p>", "\n\n")
        text = unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        lines.extend([f"### **{user}** commented:", "", text, "", "---", ""])

    return "\n".join(lines).rstrip() + "\n"


class FetchTopStoriesEdge(Edge):
    async def pre_process(self, data, settings):
        return await fetch_hn_top_stories(30)

    def post_process(self, data, settings):
        try:
            m = re.search(r'\[.*\]', str(data), re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return json.loads(data)
        except Exception as e:
            logger.warning(f"FetchTopStoriesEdge JSON parsing failed: {e}")
            return []


class FilterEdge(Edge):
    def post_process(self, data, settings):
        try:
            m = re.search(r'\[.*\]', str(data), re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return json.loads(data)
        except Exception as e:
            logger.warning(f"FilterEdge JSON parsing failed: {e}")
            return []


class FetchCommentsEdge(Edge):
    def condition(self, data, settings):
        return isinstance(data, dict) and "id" in data

    async def pre_process(self, data, settings):
        timeout = float(settings.get("timeout", 30))
        md = await fetch_hn_comments_md(data["id"], timeout=timeout)
        return {"title": data.get("title", ""), "url": data.get("url", ""), "content": md}


class SummarizeEdge(Edge):
    def pre_process(self, data, settings):
        if isinstance(data, dict):
            title = data.get("title", "Unknown")
            url = data.get("url", "")
            content = data.get("content", "")
            return f"Title: {title}\nURL: {url}\n\n{content}"
        return str(data)


class ProcessStoriesMap(MapEdge):
    """
    Subclass of MapEdge used to concurrently process the list of stories.
    Pipeline steps are defined in config.json.
    """
    pass
