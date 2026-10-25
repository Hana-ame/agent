"""HN AI Report V4 — Edge functions for the V4 vertex-edge pipeline.

Migration from legacy MapEdge architecture:
- Old: MapEdge fan-out (FetchCommentsEdge + SummarizeEdge per story)
- V4:  Single CodeEdgeV4 that internally loops through stories

Usage as CodeEdgeV4 scripts:
    fetch_stories(content, settings, staging)  → JSON array of top stories
    summarize_all(content, settings, staging)  → Final markdown report
"""

import asyncio
import json
import logging
import re
from html import unescape
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HN_API = "https://hacker-news.firebaseio.com/v0"
_HEADERS = {"User-Agent": "Mozilla/5.0 (V4-Example)"}
_MAX_COMMENTS = 15

AI_FILTER_PROMPT = """\
Here is a list of top Hacker News stories in JSON format:
{data}

Task: Select stories related to AI, LLMs, machine learning, and deep learning.
Select all relevant candidates without restricting count.

Output only a valid JSON array whose elements contain 'id', 'title', and 'url' fields.
Do not include markdown formatting or explanations.
"""

SUMMARIZE_PROMPT = """\
You are an AI topics observer on Hacker News. Summarize this story's discussion \
into a concise briefing.

## Guidelines

1. **Body only**: Return markdown organized into three sections: \
[Discussion Points], [Technical Details], and [Community Perspectives].
2. **Concise and informative**: Use minimal bullet points with high signal-to-noise ratio. \
Avoid filler sentences.
3. **Faithful to source**: Only include models, tools, links, users, and arguments \
actually present in the thread. Do not hallucinate.
4. **Language**: English.
5. **No redundant titles/links**: Do not repeat the thread title or URL.

## Story
{content}
"""


# ---------------------------------------------------------------------------
# Internal helpers (shared by fetch and summarize functions)
# ---------------------------------------------------------------------------

async def _http_client(proxy: Optional[str] = None, timeout: float = 30) -> httpx.AsyncClient:
    """Create an httpx AsyncClient with proxy support."""
    kwargs: Dict[str, Any] = {"headers": _HEADERS, "timeout": timeout, "trust_env": True}
    if proxy:
        kwargs["proxy"] = proxy
    return httpx.AsyncClient(**kwargs)


async def _fetch_story_items(
    client: httpx.AsyncClient,
    story_ids: List[int],
) -> List[Dict]:
    """Fetch HN story items by ID concurrently."""

    async def _fetch_one(item_id: int) -> Optional[Dict]:
        try:
            r = await client.get(f"{_HN_API}/item/{item_id}.json")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning("Failed to fetch story %d: %s", item_id, e)
            return None

    items = await asyncio.gather(*(_fetch_one(i) for i in story_ids))
    return [item for item in items if item and item.get("type") == "story"]


async def _fetch_comments_md(
    client: httpx.AsyncClient,
    story_id: int,
    max_comments: int = _MAX_COMMENTS,
) -> str:
    """Fetch top comments for a story and format as markdown."""
    try:
        r = await client.get(f"{_HN_API}/item/{story_id}.json")
        r.raise_for_status()
        story = r.json()
    except Exception as e:
        logger.warning("Failed to fetch story %d: %s", story_id, e)
        return "_Story not found._"

    if not story:
        return "_Story not found._"

    kids = story.get("kids", [])[:max_comments]
    if not kids:
        return "_No comments._"

    async def _fetch_comment(item_id: int) -> Optional[Dict]:
        try:
            res = await client.get(f"{_HN_API}/item/{item_id}.json")
            res.raise_for_status()
            return res.json()
        except Exception as e:
            logger.warning("Failed to fetch comment %d: %s", item_id, e)
            return None

    comments = await asyncio.gather(*(_fetch_comment(i) for i in kids))

    lines = [
        f"# {story.get('title', 'Unknown')}",
        "",
        f"> URL: <{story.get('url', '')}>",
        f"> HN Link: <https://news.ycombinator.com/item?id={story_id}>",
        "",
        "---",
        "",
    ]

    for c in comments:
        if not c or c.get("type") != "comment" or c.get("deleted") or c.get("dead"):
            continue
        user = c.get("by", "Unknown")
        text = c.get("text", "")
        text = text.replace("<p>", "\n\n")
        text = unescape(text)
        text = re.sub(r"<[^>]+>", "", text)
        lines.extend([f"### **{user}** commented:", "", text, "", "---", ""])

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# V4 Edge Functions (used as CodeEdgeV4 scripts)
# ---------------------------------------------------------------------------

async def fetch_stories(
    content: str,
    settings: Optional[Dict] = None,
    staging: Optional[Dict] = None,
) -> str:
    """CodeEdgeV4: Fetch top HN stories as JSON array.

    Replaces legacy FetchTopStoriesEdge.
    Input:  any trigger content (ignored)
    Output: JSON array of {id, title, url, score}
    """
    settings = settings or {}
    limit = int(settings.get("limit", 30))
    timeout = float(settings.get("timeout", 30))
    proxy = settings.get("proxy")

    async with await _http_client(proxy, timeout) as client:
        r = await client.get(f"{_HN_API}/topstories.json")
        r.raise_for_status()
        story_ids = r.json()[:limit]
        items = await _fetch_story_items(client, story_ids)

    out = [
        {
            "id": item["id"],
            "title": item.get("title", ""),
            "url": item.get("url", f"https://news.ycombinator.com/item?id={item['id']}"),
            "score": item.get("score", 0),
        }
        for item in items
    ]
    return json.dumps(out, ensure_ascii=False)


def filter_ai_stories_json(content: str) -> str:
    """Parse and return the JSON array from LLM filter output.

    Used by LLMEdgeV4 output handling — the LLM returns a JSON array
    of {id, title, url} objects. This is a passthrough since LLMEdgeV4
    with `json` attribute already validates JSON syntax.
    """
    return content


async def summarize_all(
    content: str,
    settings: Optional[Dict] = None,
    staging: Optional[Dict] = None,
) -> str:
    """CodeEdgeV4: Fetch comments + LLM-summarize each AI story in parallel.

    Replaces legacy ProcessStoriesMap (MapEdge fan-out).
    Input:  JSON array of {id, title, url}
    Output: Final markdown report

    Settings:
        model:       LLM model name
        base_url:    LLM API base URL
        max_conc:    Max concurrent summaries (default 1)
        proxy:       HTTP proxy for HN API
        api_key:     LLM API key
    """
    settings = settings or {}
    max_conc = int(settings.get("max_conc", 1))
    proxy = settings.get("proxy")
    api_key = settings.get("api_key", "")
    base_url = settings.get("base_url", "https://sensenova.moonchan.xyz/v1/chat/completions")
    model = settings.get("model", "sensenova-6.8-flash-lite")
    timeout = float(settings.get("timeout", 7200))

    try:
        stories = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return "_No valid AI stories found._"

    if not isinstance(stories, list) or len(stories) == 0:
        return "_No AI stories to summarize._"

    sem = asyncio.Semaphore(max_conc)

    # --- Stage 1: Fetch comments for all stories (parallel, no semaphore) ---
    async def _fetch_one(story: Dict) -> Dict:
        async with await _http_client(proxy, 30) as client:
            md = await _fetch_comments_md(client, story["id"])
        return {
            "id": story["id"],
            "title": story.get("title", "Unknown"),
            "url": story.get("url", ""),
            "content": md,
        }

    fetched = await asyncio.gather(
        *(_fetch_one(s) for s in stories)
    )

    # --- Stage 2: LLM-summarize each story (parallel, with semaphore) ---
    async def _summarize_one(fetched_story: Dict) -> str:
        async with sem:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            prompt = SUMMARIZE_PROMPT.replace("{content}", fetched_story["content"])
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000,
            }

            async with httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                trust_env=True,
            ) as client:
                r = await client.post(base_url, json=payload)
                r.raise_for_status()
                data = r.json()

            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return str(data)

    summaries = await asyncio.gather(
        *(_summarize_one(fs) for fs in fetched),
        return_exceptions=True,
    )

    # --- Stage 3: Assemble final report ---
    report_lines = [
        "# 🤖 HN AI Report — V4 Pipeline",
        f"",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        f"*Stories: {len(stories)} | Pipeline: V4 Vertex-Edge Framework*",
        "",
        "---",
        "",
    ]

    for i, (fs, summary) in enumerate(zip(fetched, summaries), 1):
        if isinstance(summary, Exception):
            report_lines.append(
                f"## {i}. {fs['title']}",
                "",
                f"> [View on HN](https://news.ycombinator.com/item?id={fs['id']})",
                "",
                f"_Failed: {summary}_",
                "",
                "---",
                "",
            )
            continue

        report_lines.append(f"## {i}. {fs['title']}")
        report_lines.append("")
        report_lines.append(f"> [Story]({fs['url']}) | [Discussion](https://news.ycombinator.com/item?id={fs['id']})")
        report_lines.append("")
        report_lines.append(summary)
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    return "\n".join(report_lines)
