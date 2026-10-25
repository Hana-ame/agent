"""v9 推理循环：现有知识 → 候选关系 → 外部证据 → 新 claim。

generate(prompt) -> 文本；fetch(entity) -> {"url","text"}。
没有外部来源或推理链的 claim 不能入库。
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse
from typing import Any, Callable, Dict, List, Optional

from v9.knowledge import Knowledge

PROMPT = """你是开放世界知识推理器。只根据给出的外部证据，判断两个实体之间是否存在一条可核验的关系。

实体 A：{a}
证据 A（{url_a}）：
{text_a}

实体 B：{b}
证据 B（{url_b}）：
{text_b}

只输出 JSON，不要 Markdown 代码块：
{{"relation": "none" 或 "有关系的谓词，如 uses/part_of/contrasts_with",
  "statement": "一句可核验的中文事实，不超过 80 字",
  "confidence": 0.0,
  "evidence_url": "必须是上面两个 URL 之一"}}

如果证据不足，relation 输出 "none"，不要编造。"""


def normalize_source(url: str, fetched: tuple[str, ...]) -> Optional[str]:
    """允许 Wikipedia 人类页面 ↔ API summary 互相映射；其余必须精确匹配。"""
    if url in fetched:
        return url

    def title(u: str) -> str:
        path = urlparse(u).path.rstrip("/")
        return path.rsplit("/", 1)[-1].replace("_", " ").lower()

    for cand in fetched:
        if ("wikipedia.org" in url and "wikipedia.org" in cand
                and title(url) == title(cand)):
            return cand
    return None


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start:end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


class InferenceEngine:
    def __init__(self, knowledge: Knowledge, *,
                 generate: Callable[[str], str],
                 fetch: Callable[[str], Dict[str, str]]) -> None:
        self.k = knowledge
        self.generate = generate
        self.fetch = fetch
        self._cache: Dict[str, Dict[str, str]] = {}

    def _evidence(self, entity: str) -> Dict[str, str]:
        if entity in self._cache:
            return self._cache[entity]
        ev = self.fetch(entity)
        if not ev.get("url") or not ev.get("text"):
            raise ValueError(f"fetch({entity!r}) 没有返回 url/text")
        self._cache[entity] = ev
        if self.k.get_source(ev["url"]) is None:
            self.k.put_source(ev["url"], ev["text"])
        return ev

    def step(self) -> Dict[str, Any]:
        pairs = self.k.unexplored_pairs(1)
        if not pairs:
            return {"status": "no_pairs"}
        a, b = pairs[0]
        ea, eb = self._evidence(a), self._evidence(b)
        raw = self.generate(PROMPT.format(
            a=a, b=b, url_a=ea["url"], url_b=eb["url"],
            text_a=ea["text"][:1200], text_b=eb["text"][:1200]))
        data = parse_json(raw) or {}
        relation = str(data.get("relation") or "none").strip()
        statement = str(data.get("statement") or "").strip()
        if relation == "none" or not statement:
            self.k.mark_explored(a, b, "none")
            return {"status": "no_relation", "pair": [a, b], "raw": raw[:200]}
        if len(statement) > 200:
            self.k.mark_explored(a, b, "too_long")
            return {"status": "rejected", "pair": [a, b],
                    "reason": "statement 超过 200 字"}

        raw_url = str(data.get("evidence_url") or "").strip()
        url = normalize_source(raw_url, (ea["url"], eb["url"]))
        if url is None:
            self.k.mark_explored(a, b, "bad_source")
            return {"status": "rejected", "pair": [a, b],
                    "reason": "evidence_url 不在抓取来源里"}

        cid = self.k.add_claim(
            a, relation, b, statement,
            confidence=float(data.get("confidence", 0.5)),
            source_url=url)
        self.k.mark_explored(a, b, "linked")
        return {"status": "claim", "pair": [a, b], "claim_id": cid,
                "relation": relation, "statement": statement, "source": url}

    def run(self, cycles: int = 3) -> List[Dict[str, Any]]:
        out = []
        for _ in range(cycles):
            res = self.step()
            out.append(res)
            if res["status"] in ("no_pairs",):
                break
        return out
