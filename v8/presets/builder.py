"""把 N 条预设 + 一段会话状态，构造成一张 v8 DAG：

    state ─→ check_a (pi) ─┐
    state ─→ check_b (pi) ─┼→ aggregate (script) ─→ result
    state ─→ check_c (pi) ─┘

每条预设 = 一个 pi 节点（完整 system prompt + 状态），全部并行检查；
聚合是确定性 script，不做 LLM 总结。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .presets import Preset

DEFAULT_PI: Dict[str, Any] = {
    "provider": "laptop",
    "model": "sensenova-6.8-flash-lite",
    "tools": False,
    "thinking": "off",
    "timeout": 300,
    "retries": 2,
}

PROMPT = """{system_prompt}

【会话状态】
{{tag:state}}

【输出要求】
只输出 JSON，不要 Markdown 代码块、不要解释：
{{"satisfied": true 或 false, "score": 0-5, "reason": "一段判断理由"}}
"""

AGGREGATE_SCRIPT = str(Path(__file__).resolve().parent / "scripts" / "aggregate.py")


def build_dag(presets: List[Preset], *, pi: Dict[str, Any] | None = None,
              name: str = "preset_check") -> Dict[str, Any]:
    pi = dict(DEFAULT_PI, **(pi or {}))
    edges: List[Dict[str, Any]] = []
    for p in presets:
        edges.append({
            "id": p.tag,
            "kind": "pi",
            "layer": 0,
            "input_tags": ["state"],
            "output_tags": [p.tag],
            "prompt": PROMPT.format(system_prompt=p.system_prompt.strip()),
        })
    edges.append({
        "id": "aggregate",
        "kind": "script",
        "layer": 1,
        "input_tags": [p.tag for p in presets],
        "output_tags": ["result"],
        "script": AGGREGATE_SCRIPT,
        "timeout": 60,
    })
    return {
        "name": name,
        "seed_tag": "state",
        "result_tag": "result",
        "pi": pi,
        "edges": edges,
    }
