"""LLM & Embedding Client layer for DAG Agent.

Supports OpenAI-compatible HTTP endpoints configured via `models.json` (`{key: {endpoint, apikey, ...}}`)
as well as a deterministic `MockLLMClient` for offline testing and verification of clean-context requests.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

from dag.schema import ModelConfig

# 重试白名单（自 v6 线移植，2026-10-22）。
# 本机网关 laptop-8000 / sensenova 会在 200 与 403(WAF code 1010) 之间间歇抖动，
# 403 是瞬时封锁不是鉴权失败；一次长任务几十次连续调用必撞。必须重试。
# 10054 是 Windows 连接重置，也属于瞬时。
_RETRYABLE_MARKERS: Tuple[str, ...] = (
    "429", "rate_limit", "Timeout", "timed out",
    "ConnectError", "ReadError", "10054", "reset",
    "502", "503", "504", "403", "1010",
)


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
    """Production client that calls `{endpoint}/chat/completions` and `{endpoint}/embeddings` using `models.json`.

    两个自 v6 线移植的能力（2026-10-22）：

    - **SSE 流式**（`stream=True`）：非流式请求在长输出时会长时间无字节返回，被中间层
      掐断；流式有增量帧，稳定性高一个量级。流式下若 `content` 全空则回落到
      `reasoning` / `reasoning_content`（推理模型常把思考放这两处）。
    - **重试白名单**：429 / 403(1010) / 5xx / 超时 / 连接重置 都退避重试，
      鉴权与参数错误直接抛，不浪费配额。
    """

    def __init__(
        self,
        timeout: float = 120.0,
        fallback_embedding_dim: int = 32,
        *,
        max_retries: int = 3,
        stream: bool = False,
        sse_timeout: float = 900.0,
        transport: Any = None,
    ) -> None:
        self.timeout = timeout
        self.fallback_embedding_dim = fallback_embedding_dim
        self.max_retries = max(1, int(max_retries))
        self.stream = stream
        self.sse_timeout = sse_timeout
        self.transport = transport  # httpx transport，测试注入 MockTransport
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
        if "max_tokens" not in body:
            body["max_tokens"] = 1024

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "dag-agent/1.0",
        }
        if cfg.apikey and cfg.apikey != "not-needed":
            headers["Authorization"] = f"Bearer {cfg.apikey}"
        if self.stream:
            headers["Accept"] = "text/event-stream"

        last_err: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                if self.stream:
                    text = self._stream_once(url, headers, body)
                else:
                    text = self._blocking_once(url, headers, body)
                if not text.strip():
                    raise RuntimeError(f"空响应（attempt={attempt + 1}）")
                return text
            except BaseException as exc:  # noqa: BLE001 - 统一判定是否值得重试
                last_err = exc
                msg = f"{type(exc).__name__}: {exc}"
                retryable = any(tag in msg for tag in _RETRYABLE_MARKERS)
                if not retryable or attempt == self.max_retries - 1:
                    raise
                # 指数退避，上限 8s：3 次重试最多多等 ~1.5s（0.5+1.0）
                time.sleep(min(8.0, 0.5 * (2 ** attempt)))
        raise RuntimeError(f"LLM 调用失败: {last_err}")  # pragma: no cover - 循环必 return/raise

    def _blocking_once(self, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> str:
        try:
            import requests

            resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except ImportError:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Empty choices from {url}: {data}")
        msg = choices[0].get("message") or {}
        # Priority: content first, then reasoning_content, reasoning, text
        for field_name in ("content", "reasoning_content", "reasoning", "text"):
            val = msg.get(field_name)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    def _stream_once(self, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> str:
        """消费 SSE 流，拼出正文；正文全空时回落到 reasoning 通道。"""
        import httpx

        out: List[str] = []
        reason: List[str] = []
        payload = dict(body, stream=True)
        # trust_env=False：本机 NO_PROXY 里若有 "[::1]" 之类非法项，httpx 解析代理会抛 InvalidURL
        with httpx.Client(
            timeout=httpx.Timeout(self.sse_timeout),
            trust_env=False,
            transport=self.transport,
        ) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    raw = resp.read().decode("utf-8", "replace")[:500]
                    raise RuntimeError(f"HTTP {resp.status_code}: {raw}")
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk_raw = line[5:].strip()
                    if chunk_raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_raw)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if isinstance(delta.get("content"), str):
                        out.append(delta["content"])
                    elif isinstance(delta.get("reasoning"), str):
                        reason.append(delta["reasoning"])
                    elif isinstance(delta.get("reasoning_content"), str):
                        reason.append(delta["reasoning_content"])
        text = "".join(out)
        return text if text.strip() else "".join(reason)

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
