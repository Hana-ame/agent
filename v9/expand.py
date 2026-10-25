"""从外部页面链接扩展实体（开放世界，不封闭 seed）。"""
from __future__ import annotations

import json
import urllib.parse
from typing import Callable, Dict, List

from v9.knowledge import Knowledge

API = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
       "&prop=links&pllimit=30&titles={title}")


def wiki_url(title: str) -> str:
    return ("https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(title.replace(" ", "_")))


def expand(knowledge: Knowledge, fetch: Callable[[str], Dict[str, str]],
           entity: str, limit: int = 3) -> List[str]:
    """抓 entity 页面的外链，挑前 limit 个新实体写回；返回新增名字。"""
    url = API.format(title=urllib.parse.quote(entity.replace(" ", "_")))
    data = json.loads(fetch(url)["text"])
    pages = (data.get("query") or {}).get("pages") or {}
    links: List[str] = []
    for page in pages.values():
        for item in page.get("links") or []:
            title = str(item.get("title") or "").strip()
            if title and title not in links:
                links.append(title)
    known = {e["name"] for e in knowledge.entities()}
    added: List[str] = []
    for title in links:
        if title in known or len(added) >= limit:
            continue
        knowledge.add_entity(title, source_url=wiki_url(title))
        added.append(title)
    return added
