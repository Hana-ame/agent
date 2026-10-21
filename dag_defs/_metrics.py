"""评测/浸泡两类 DAG 共用的指标工具。

放在 `_` 开头文件名下：`load_definitions` 会跳过它，不会被当成任务 DAG 注册。
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List

ERROR_MARKERS = ("Traceback (most recent call last)", "Error:", "Exception:", "\x00")


def as_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, (list, tuple)):
        return "\n\n".join(as_text(x) for x in payload)
    contents = getattr(payload, "contents", None)
    if contents is not None:
        return contents if isinstance(contents, str) else json.dumps(contents, ensure_ascii=False)
    return str(payload)


def structure(text: str) -> Dict[str, Any]:
    """报告的结构指纹。用于跨轮比较「形状稳不稳」。"""
    lines = text.splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    return {
        "chars": len(text),
        "lines": len(lines),
        "nonempty_lines": len(nonempty),
        "sections": len(re.findall(r"^#{1,4} ", text, re.M)),
        "ends_clean": bool(re.search(r"[。！？.!)\]》”\"]\s*$", text.rstrip())),
        "has_error": any(m in text for m in ERROR_MARKERS),
    }


def ngrams(text: str, n: int = 3) -> set:
    t = re.sub(r"\s+", "", text)
    return {t[i : i + n] for i in range(max(0, len(t) - n + 1))}


def jaccard(a: str, b: str) -> float:
    sa, sb = ngrams(a), ngrams(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def agreement(values: List[Any]) -> float:
    """众数占比：1.0 = 所有取值相同。"""
    if not values:
        return 0.0
    best = max(values.count(v) for v in set(values))
    return best / len(values)


def p95(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(0.95 * len(s)) - 1))
    return s[idx]


def pair_jaccard(texts: List[str]) -> float:
    """两两 Jaccard 的均值；不足两份返回 1.0。"""
    pairs = [jaccard(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))]
    return sum(pairs) / len(pairs) if pairs else 1.0
