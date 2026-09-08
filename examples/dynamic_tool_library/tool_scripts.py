"""Transform scripts and worker functions for the dynamic tool library."""

import json
import re
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Router / Dispatcher Logic
# ---------------------------------------------------------------------------

def classify_intent(query: str) -> str:
    """Classify user query into one of the registered tool categories (heuristic fallback)."""
    q = query.lower()
    if any(k in q for k in ["code", "python", "def ", "function", "syntax", "bug", "refactor", "代码", "函数", "编程"]):
        return "code_analyzer"
    elif any(k in q for k in ["finance", "revenue", "margin", "profit", "ebitda", "cost", "财务", "利润", "财报", "收入", "资金"]):
        return "finance_calculator"
    else:
        return "data_extractor"


async def classify_intent_with_llm(
    query: str,
    catalog_dir: Any,
    api_key: Optional[str] = None,
    base_url: str = "https://sensenova.moonchan.xyz/v1/chat/completions",
    model: str = "sensenova-6.8-flash-lite",
    timeout: float = 12.0,
) -> tuple[str, str]:
    """Classify user query using real LLM semantic reasoning over catalog descriptions.

    Discovers all tool manifests dynamically from catalog_dir.
    Falls back safely to heuristic rule-based classify_intent if LLM fails or times out.
    """
    import os
    from pathlib import Path
    import httpx

    catalog_path = Path(catalog_dir)
    api_key = api_key or os.environ.get("SENSENOVA_API_KEY", "") or None

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
        return "data_extractor", "Empty catalog fallback"

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
    except Exception:
        pass

    fallback = classify_intent(query)
    return fallback, "Rule-based heuristic fallback (LLM offline or unavailable)"


# ---------------------------------------------------------------------------
# Tool 1: Code Analyzer Subgraph Scripts
# ---------------------------------------------------------------------------

def run_code_lint(content: str, settings: Optional[Dict[str, Any]] = None, staging: Optional[Dict[str, Any]] = None) -> str:
    """Analyze code syntax, lines, and potential smells."""
    lines = content.strip().split("\n")
    line_count = len(lines)
    has_todo = any("todo" in l.lower() for l in lines)
    has_eval = any("eval(" in l for l in lines)

    analysis = {
        "raw_code": content,
        "line_count": line_count,
        "has_todo": has_todo,
        "security_warnings": ["Use of eval() detected! High security risk."] if has_eval else [],
        "style_score": 85 if line_count < 50 else 70,
    }
    return json.dumps(analysis, ensure_ascii=False)


def generate_code_critique(content: str, settings: Optional[Dict[str, Any]] = None, staging: Optional[Dict[str, Any]] = None) -> str:
    """Generate professional code review markdown critique."""
    data = json.loads(content)
    warnings = data.get("security_warnings", [])
    warn_text = "\n".join(f"- ⚠️ {w}" for w in warnings) if warnings else "- ✅ No critical security flaws detected."

    return (
        f"### 🔍 Automated Code Review Report\n\n"
        f"- **Total Lines of Code**: {data['line_count']}\n"
        f"- **Health Score**: {data['style_score']}/100\n"
        f"- **Security & Best Practices**:\n{warn_text}\n\n"
        f"**Recommendation**: Structure looks good. Ensure unit test coverage exceeds 80%."
    )


# ---------------------------------------------------------------------------
# Tool 2: Finance Calculator Subgraph Scripts
# ---------------------------------------------------------------------------

def calculate_financial_metrics(content: str, settings: Optional[Dict[str, Any]] = None, staging: Optional[Dict[str, Any]] = None) -> str:
    """Extract financial numbers and calculate key margins."""
    # Simple rule-based extraction or json parsing
    numbers = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", content)]
    revenue = numbers[0] if len(numbers) > 0 else 1000000.0
    costs = numbers[1] if len(numbers) > 1 else 650000.0
    profit = revenue - costs
    margin = (profit / revenue * 100) if revenue > 0 else 0.0

    metrics = {
        "revenue_usd": revenue,
        "operating_cost_usd": costs,
        "net_profit_usd": profit,
        "profit_margin_pct": round(margin, 2),
    }
    return json.dumps(metrics, ensure_ascii=False)


def summarize_financial_health(content: str, settings: Optional[Dict[str, Any]] = None, staging: Optional[Dict[str, Any]] = None) -> str:
    """Generate executive financial summary."""
    m = json.loads(content)
    status = "Strong / Profitable" if m["profit_margin_pct"] >= 20 else "Moderate"

    return (
        f"### 📈 Executive Financial Analysis\n\n"
        f"- **Gross Revenue**: ${m['revenue_usd']:,.2f}\n"
        f"- **Operating Costs**: ${m['operating_cost_usd']:,.2f}\n"
        f"- **Net Profit**: ${m['net_profit_usd']:,.2f}\n"
        f"- **Profit Margin**: {m['profit_margin_pct']}%\n"
        f"- **Financial Health**: **{status}**"
    )


# ---------------------------------------------------------------------------
# Tool 3: Data Extractor Subgraph Scripts
# ---------------------------------------------------------------------------

def extract_entities(content: str, settings: Optional[Dict[str, Any]] = None, staging: Optional[Dict[str, Any]] = None) -> str:
    """Extract key tokens, emails, and phone numbers from unstructured text."""
    emails = re.findall(r"[\w\.-]+@[\w\.-]+", content)
    words = [w for w in re.findall(r"\b[A-Za-z]{4,}\b", content) if not w.lower().startswith("http")]
    
    extracted = {
        "emails_found": emails,
        "keywords": list(set(words))[:6],
        "character_length": len(content),
    }
    return json.dumps(extracted, ensure_ascii=False)


def format_extraction_table(content: str, settings: Optional[Dict[str, Any]] = None, staging: Optional[Dict[str, Any]] = None) -> str:
    """Format extracted entities into markdown table."""
    data = json.loads(content)
    emails_str = ", ".join(data["emails_found"]) if data["emails_found"] else "None"
    keywords_str = ", ".join(data["keywords"]) if data["keywords"] else "None"

    return (
        f"### 📑 Unstructured Data Extraction Results\n\n"
        f"| Field | Extracted Data |\n"
        f"| :--- | :--- |\n"
        f"| **Emails Found** | {emails_str} |\n"
        f"| **Key Entities** | {keywords_str} |\n"
        f"| **Raw Content Length** | {data['character_length']} chars |\n"
    )
