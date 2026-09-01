"""Telemetry, Token Tracking & Cost Profiling module.

Tracks prompt tokens, completion tokens, latency, and estimated cost across
individual edge executions and entire graph workflows.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("vertex_edge_agent.telemetry")


# Standard Pricing Catalog (USD per 1,000,000 tokens)
DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    # model -> {"prompt_per_m": ..., "completion_per_m": ...}
    "gemini-1.5-pro": {"prompt_per_m": 3.50, "completion_per_m": 10.50},
    "gemini-1.5-flash": {"prompt_per_m": 0.075, "completion_per_m": 0.30},
    "gemini-2.0-flash": {"prompt_per_m": 0.10, "completion_per_m": 0.40},
    "gpt-4o": {"prompt_per_m": 2.50, "completion_per_m": 10.00},
    "gpt-4o-mini": {"prompt_per_m": 0.15, "completion_per_m": 0.60},
    "claude-3-5-sonnet": {"prompt_per_m": 3.00, "completion_per_m": 15.00},
    # Free-tier models: no API key, zero cost — listed explicitly so free-tier
    # report runs are NOT billed at the "default" rates above.
    "hy3-free": {"prompt_per_m": 0.0, "completion_per_m": 0.0},
    # Models not listed here (e.g. sensenova-*) fall back to "default" rates —
    # the estimate is a placeholder for the report, not a billable figure.
    "default": {"prompt_per_m": 1.00, "completion_per_m": 3.00},
}


@dataclass
class UsageMetrics:
    """Token usage and cost metrics for a single execution or aggregated workflow."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    def add(self, other: "UsageMetrics") -> "UsageMetrics":
        return UsageMetrics(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_ms=self.latency_ms + other.latency_ms,
        )

    def to_dict(self) -> Dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 2),
        }


def estimate_tokens(text: Optional[str]) -> int:
    """Rough estimation of token count (heuristic: ~4 characters per token)."""
    if not text:
        return 0
    return max(1, len(str(text)) // 4)


def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str = "default") -> float:
    """Compute estimated USD cost based on token counts and model pricing."""
    rates = DEFAULT_PRICING.get(model, DEFAULT_PRICING["default"])
    p_cost = (prompt_tokens / 1_000_000.0) * rates["prompt_per_m"]
    c_cost = (completion_tokens / 1_000_000.0) * rates["completion_per_m"]
    return p_cost + c_cost


class TelemetryTracker:
    """Collects and aggregates per-edge and workflow-wide telemetry."""

    def __init__(self):
        self.edge_metrics: Dict[str, UsageMetrics] = {}

    def record_edge(
        self,
        edge_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        latency_ms: float,
    ) -> UsageMetrics:
        """Record usage metrics for an edge execution."""
        cost = calculate_cost(prompt_tokens, completion_tokens, model)
        total = prompt_tokens + completion_tokens
        metrics = UsageMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
        self.edge_metrics[edge_id] = metrics
        return metrics

    def get_total_metrics(self) -> UsageMetrics:
        """Aggregate metrics across all tracked edges."""
        total = UsageMetrics()
        for m in self.edge_metrics.values():
            total = total.add(m)
        return total
