"""LLM & Embedding Client layer for DAG Agent.

Supports OpenAI-compatible HTTP endpoints configured via `models.json` (`{key: {endpoint, apikey, ...}}`)
as well as a deterministic `MockLLMClient` for offline testing and verification of clean-context requests.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union, runtime_checkable

from dag_agent.schema import ModelConfig


@dataclass
class LLMCallRecord:
    """Audit record of a single LLM request, capturing whether its context is clean."""

    phase: str  # "edge" | "default_routing" | "default_clean_output" | "meta"
    prompt: str
    system_prompt: str
    model_key: str
    endpoint: str
    is_clean_context: bool
    selected_inputs: List[str] = field(default_factory=list)


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM completion and embedding providers."""

    def complete(
        self,
        prompt: str,
        *,
        model_config: Optional[ModelConfig] = None,
        system_prompt: str = "",
        phase: str = "edge",
        is_clean_context: bool = True,
        selected_inputs: Optional[Sequence[str]] = None,
    ) -> str: ...

    def embed(self, text: str, *, model_config: Optional[ModelConfig] = None) -> List[float]: ...


def deterministic_semantic_embedding(text: str, dim: int = 32) -> List[float]:
    """Compute a deterministic character/subword semantic vector for offline & fallback embedding similarity."""
    cleaned = str(text).lower().strip()
    if not cleaned:
        return [0.0] * dim
    synonym_groups = [
        ({"revenue", "income", "sales", "营收", "收入", "销售额"}, "sem_revenue"),
        ({"cost", "expense", "expenditure", "成本", "支出", "费用"}, "sem_cost"),
        ({"profit", "net_profit", "margin", "利润", "净利润"}, "sem_profit"),
        ({"report", "raw_report", "statement", "财报", "报告", "报表"}, "sem_report"),
        ({"user", "customer", "profile", "用户", "客户", "画像"}, "sem_user"),
        ({"order", "purchase", "transaction", "订单", "交易"}, "sem_order"),
        ({"query", "question", "input", "提问", "问题", "输入"}, "sem_query"),
        ({"summary", "conclusion", "result", "汇总", "结论", "结果"}, "sem_result"),
        ({"code", "source", "script", "代码", "源码"}, "sem_code"),
        ({"review", "audit", "check", "审核", "检查", "校验"}, "sem_review"),
    ]
    tokens = [cleaned]
    for group, canonical in synonym_groups:
        for word in group:
            if word in cleaned:
                tokens.append(canonical)
    for ch in cleaned:
        if ch.strip() and ch not in "_-:":
            tokens.append(f"c:{ch}")
    for i in range(len(cleaned) - 1):
        bg = cleaned[i : i + 2]
        if bg.strip():
            tokens.append(f"bg:{bg}")

    vec = [0.0] * dim
    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        weight = 3.0 if tok.startswith("sem_") else 1.0
        for d in range(dim):
            bit = 1.0 if (h[d % len(h)] & (1 << (d % 8))) else -1.0
            vec[d] += weight * bit

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class OpenAICompatibleClient:
    """Production client that calls `{endpoint}/chat/completions` and `{endpoint}/embeddings` using `models.json`."""

    def __init__(self, timeout: float = 120.0, fallback_embedding_dim: int = 32) -> None:
        self.timeout = timeout
        self.fallback_embedding_dim = fallback_embedding_dim
        self.call_history: List[LLMCallRecord] = []

    def complete(
        self,
        prompt: str,
        *,
        model_config: Optional[ModelConfig] = None,
        system_prompt: str = "",
        phase: str = "edge",
        is_clean_context: bool = True,
        selected_inputs: Optional[Sequence[str]] = None,
    ) -> str:
        cfg = model_config or ModelConfig(key="default")
        self.call_history.append(
            LLMCallRecord(
                phase=phase,
                prompt=prompt,
                system_prompt=system_prompt,
                model_key=cfg.key,
                endpoint=cfg.endpoint,
                is_clean_context=is_clean_context,
                selected_inputs=list(selected_inputs or []),
            )
        )
        url = cfg.endpoint.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "model": cfg.model or cfg.key,
            "messages": messages,
            **cfg.params,
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "dag-agent/1.0",
        }
        if cfg.apikey and cfg.apikey != "not-needed":
            headers["Authorization"] = f"Bearer {cfg.apikey}"

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Empty choices from {url}: {data}")
        msg = choices[0].get("message") or {}
        for field_name in ("content", "reasoning_content", "reasoning", "text"):
            val = msg.get(field_name)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    def embed(self, text: str, *, model_config: Optional[ModelConfig] = None) -> List[float]:
        if model_config and model_config.params.get("use_remote_embedding"):
            url = model_config.endpoint.rstrip("/")
            if not url.endswith("/embeddings"):
                url = f"{url}/embeddings"
            headers = {"Content-Type": "application/json"}
            if model_config.apikey and model_config.apikey != "not-needed":
                headers["Authorization"] = f"Bearer {model_config.apikey}"
            body = {"model": model_config.model, "input": text}
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return list(data["data"][0]["embedding"])
        return deterministic_semantic_embedding(text, dim=self.fallback_embedding_dim)


class MockLLMClient:
    """Deterministic LLM & Embedding client for testing and local verification.

    Records every call in `self.call_history` so tests can inspect exact prompts,
    model keys, and verify clean-context isolation.
    """

    def __init__(
        self,
        rules: Optional[Dict[str, Union[str, Callable[..., Any]]]] = None,
        model_handlers: Optional[Dict[str, Union[str, Callable[..., Any]]]] = None,
        default_handler: Optional[Callable[[str, Optional[ModelConfig]], str]] = None,
        custom_embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        self.rules: Dict[str, Union[str, Callable[..., Any]]] = dict(rules or {})
        self.model_handlers: Dict[str, Union[str, Callable[..., Any]]] = dict(model_handlers or {})
        self.default_handler = default_handler
        self.custom_embeddings: Dict[str, List[float]] = dict(custom_embeddings or {})
        self.call_history: List[LLMCallRecord] = []

    def add_rule(self, keyword: str, response: Union[str, Callable[..., Any]]) -> None:
        self.rules[keyword] = response

    def complete(
        self,
        prompt: str,
        *,
        model_config: Optional[ModelConfig] = None,
        system_prompt: str = "",
        phase: str = "edge",
        is_clean_context: bool = True,
        selected_inputs: Optional[Sequence[str]] = None,
    ) -> str:
        cfg = model_config or ModelConfig(key="default")
        self.call_history.append(
            LLMCallRecord(
                phase=phase,
                prompt=prompt,
                system_prompt=system_prompt,
                model_key=cfg.key,
                endpoint=cfg.endpoint,
                is_clean_context=is_clean_context,
                selected_inputs=list(selected_inputs or []),
            )
        )
        for kw, handler in self.rules.items():
            if kw in prompt or (system_prompt and kw in system_prompt):
                return self._invoke(handler, prompt, cfg)
        if cfg.key in self.model_handlers:
            return self._invoke(self.model_handlers[cfg.key], prompt, cfg)
        if cfg.model in self.model_handlers:
            return self._invoke(self.model_handlers[cfg.model], prompt, cfg)
        if self.default_handler is not None:
            return self.default_handler(prompt, cfg)
        return f"[{cfg.key}] {prompt.strip()}"

    @staticmethod
    def _invoke(handler: Union[str, Callable[..., Any]], prompt: str, cfg: ModelConfig) -> str:
        if callable(handler):
            try:
                res = handler(prompt, cfg)
            except TypeError:
                res = handler(prompt)
            return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
        return str(handler)

    def embed(self, text: str, *, model_config: Optional[ModelConfig] = None) -> List[float]:
        key = str(text).strip().lower()
        if key in self.custom_embeddings:
            return self.custom_embeddings[key]
        return deterministic_semantic_embedding(key)
