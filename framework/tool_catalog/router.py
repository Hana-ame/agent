"""Intent routing for the dynamic tool catalog.

Moved out of ``examples/dynamic_tool_library/tool_scripts.py`` so the server
package no longer imports from ``examples/`` (which is not shipped with the
package). The example module re-exports these names for backward compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("vertex_edge_agent.tool_catalog.router")

#: Default router endpoint / model, overridable per call or via environment.
DEFAULT_ROUTER_BASE_URL = "https://sensenova.moonchan.xyz/v1/chat/completions"
DEFAULT_ROUTER_MODEL = "sensenova-6.8-flash-lite"

#: Fallback tool id when the catalog is empty or the query matches nothing.
DEFAULT_TOOL_ID = "data_extractor"

_ROUTER_BASE_URL_ENV = "VEA_ROUTER_BASE_URL"
_ROUTER_MODEL_ENV = "VEA_ROUTER_MODEL"


def get_router_base_url() -> str:
    """Server-side router endpoint (never taken from the request body)."""
    return (os.environ.get(_ROUTER_BASE_URL_ENV) or "").strip() or DEFAULT_ROUTER_BASE_URL


def get_router_model() -> str:
    """Server-side router model id."""
    return (os.environ.get(_ROUTER_MODEL_ENV) or "").strip() or DEFAULT_ROUTER_MODEL


def classify_intent(query: str) -> str:
    """Classify user query into one of the registered tool categories (heuristic fallback)."""
    q = query.lower()
    if any(k in q for k in ["code", "python", "def ", "function", "syntax", "bug", "refactor", "代码", "函数", "编程"]):
        return "code_analyzer"
    elif any(k in q for k in ["finance", "revenue", "margin", "profit", "ebitda", "cost", "财务", "利润", "财报", "收入", "资金"]):
        return "finance_calculator"
    else:
        return DEFAULT_TOOL_ID


async def classify_intent_with_llm(
    query: str,
    catalog_dir: Any,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 12.0,
) -> tuple[str, str]:
    """Classify user query using real LLM semantic reasoning over catalog descriptions.

    Discovers all tool manifests dynamically from ``catalog_dir`` and falls back
    to the heuristic classifier if the LLM is unreachable. ``base_url``/``model``
    default to the server-side configuration so a caller cannot redirect the
    request (SSRF) or exfiltrate a key.
    """
    import httpx

    catalog_path = Path(catalog_dir)
    api_key = api_key or os.environ.get("SENSENOVA_API_KEY", "") or None
    base_url = base_url or get_router_base_url()
    model = model or get_router_model()

    # 1. Discover all tools in catalog dynamically
    tool_specs: list[str] = []
    valid_tools: set[str] = set()

    for p in sorted(catalog_path.glob("*.json")):
        tool_id = p.stem
        valid_tools.add(tool_id)
        try:
            manifest = json.loads(p.read_text(encoding="utf-8"))
            meta = manifest.get("metadata", {})
            name = meta.get("name", tool_id)
            desc = meta.get("description", "No description provided.")
            tool_specs.append(f"- {tool_id}: {name} — {desc}")
        except Exception:
            tool_specs.append(f"- {tool_id}")

    if not valid_tools:
        return DEFAULT_TOOL_ID, "Empty catalog fallback"

    tools_desc_str = "\n".join(tool_specs)

    system_prompt = (
        "You are an intelligent workflow intent classifier. "
        "Analyze the user's task request and select the single most appropriate tool from the available catalog.\n\n"
        f"Available Tools in Catalog:\n{tools_desc_str}\n\n"
        "Return strictly valid JSON with no extra commentary:\n"
        '{"tool": "<tool_id>", "reason": "<one sentence reasoning in English or Chinese>"}'
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Task Request:\n{query}"},
        ],
        "temperature": 0.1,
    }

    try:
        async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
            resp = await client.post(base_url, json=payload)
            if resp.status_code == 200:
                body = resp.json()
                content = body["choices"][0]["message"]["content"].strip()
                content = re.sub(r"^```json\s*", "", content)
                content = re.sub(r"^```\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
                parsed = json.loads(content)
                selected_tool = parsed.get("tool", "").strip()
                reason = parsed.get("reason", "Selected by LLM reasoning.")
                if selected_tool in valid_tools:
                    return selected_tool, f"LLM Match ({model}): {reason}"
    except Exception as exc:
        logger.debug("LLM intent router unavailable: %s", exc)

    fallback = classify_intent(query)
    return fallback, "Rule-based heuristic fallback (LLM offline or unavailable)"
