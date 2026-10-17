"""VEA V6 模型接口：OpenAI 兼容客户端 + 本地 `SmartMockLLM`。

要点：

- `OpenAICompatibleLLM` 支持按调用覆写 `base_url` / `api_key` / 额外参数（`dag.models` 注入）；
- 默认走 **SSE 流式**（`stream=True`）：非流式请求在长输出时会长时间无字节返回，
  被中间层掐断（实测 429 / 连接中断），流式有增量帧，稳定性高一个量级；
- 推理模型（如 sensenova-6.8-flash-lite）常把思考放 `reasoning`、`content` 为空，
  `_extract_message_text` 做 `content → reasoning_content → reasoning → text` 分级回退；
- 支持"续写"（`continue_from`）：命中长度上限时把已生成前缀回填，让模型接着写，
  这是长代码生成能落地的前提。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

__all__ = ["LLMProtocol", "OpenAICompatibleLLM", "SmartMockLLM", "extract_message_text"]


@runtime_checkable
class LLMProtocol(Protocol):
    """V6 大模型调用协议。"""

    async def ask(
        self,
        prompt: str,
        model: Optional[str] = None,
        image: Optional[str] = None,
    ) -> str: ...


def extract_message_text(data: Dict[str, Any]) -> str:
    """从 OpenAI 兼容响应体里取正文，兼容推理字段命名。"""
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"响应中没有 choices: {json.dumps(data, ensure_ascii=False)[:400]}")
    msg = choices[0].get("message") or choices[0].get("delta") or {}
    for key in ("content", "reasoning_content", "reasoning", "text"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    finish = choices[0].get("finish_reason")
    raise ValueError(
        f"模型返回空正文（finish_reason={finish}）；推理模型在小 max_tokens 下会把预算全花在思考上，"
        "请增大 max_tokens 或换非推理模型。"
    )


class LLMError(RuntimeError):
    """LLM 调用失败（网络、限流、空正文）。"""


class OpenAICompatibleLLM:
    """OpenAI 兼容端点客户端（默认流式）。"""

    def __init__(
        self,
        model: str = "sensenova-6.8-flash-lite",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("VEA_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.base_url = (
            base_url
            or os.environ.get("VEA_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.temperature = temperature
        self.timeout = timeout
        self.call_log: List[Dict[str, Any]] = []

    # ---- 打包请求 --------------------------------------------------------
    def _payload(
        self,
        prompt: str,
        model: Optional[str],
        image: Optional[str],
        max_tokens: Optional[int],
        stream: bool,
        continue_from: Optional[str],
    ) -> Tuple[str, Dict[str, Any]]:
        url = self.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"
        messages: List[Dict[str, Any]] = []
        text = prompt
        if continue_from:
            text = prompt + "\n\n【已生成的前缀，请从中断处无缝续写，不要重复前缀】\n" + continue_from
        if image:
            user_content: Any = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image}},
            ]
        else:
            user_content = text
        body: Dict[str, Any] = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": self.temperature,
            "stream": stream,
        }
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        return url, body

    # ---- 同步实现（异步入口只是 to_thread 包一层）------------------------
    def _post_sync(
        self,
        prompt: str,
        model: Optional[str] = None,
        image: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        continue_from: Optional[str] = None,
        max_retries: int = 4,
        sse_snapshot: Optional[str] = None,
        **extra: Any,
    ) -> str:
        import httpx

        use_key = api_key if api_key is not None else self.api_key
        url, body = self._payload(prompt, model, image, max_tokens, stream, continue_from)
        if base_url:
            url = base_url.rstrip("/")
            if not url.endswith("/chat/completions"):
                url = url + "/chat/completions"
        if extra.get("temperature") is not None:
            body["temperature"] = extra["temperature"]
        for k, v in extra.items():
            if k in ("temperature",):
                continue
            body[k] = v

        headers = {"Content-Type": "application/json"}
        if use_key:
            headers["Authorization"] = f"Bearer {use_key}"
        if stream:
            headers["Accept"] = "text/event-stream"

        last_err: Optional[Exception] = None
        started = time.time()
        for attempt in range(max_retries):
            try:
                if stream:
                    text, finish = self._stream_once(url, headers, body)
                else:
                    text, finish = self._blocking_once(url, headers, body)
                self.call_log.append(
                    {
                        "model": body.get("model"),
                        "chars": len(text),
                        "finish_reason": finish,
                        "seconds": round(time.time() - started, 1),
                        "attempt": attempt + 1,
                        "stream": stream,
                    }
                )
                if not text.strip():
                    raise LLMError(f"空响应（finish_reason={finish}）")
                return text
            except Exception as exc:  # noqa: BLE001 - 统一重试
                last_err = exc
                msg = str(exc)
                # 限流 / 超时 / 连接错误 → 退避重试；其它（鉴权、参数）直接抛
                retryable = any(
                    tag in msg
                    for tag in ("429", "rate_limit", "Timeout", "timed out", "ConnectError", "ReadError", "502", "503", "504")
                )
                if not retryable or attempt == max_retries - 1:
                    break
                time.sleep(min(30.0, 2.0 * (attempt + 1)))
        raise LLMError(f"LLM 调用失败: {last_err}")

    @staticmethod
    def _stream_once(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Tuple[str, str]:
        import httpx

        out: List[str] = []
        reason: List[str] = []
        finish = ""
        # trust_env=False：本机 NO_PROXY 含 "[::1]" 之类非法项会让 httpx 解析代理时抛 InvalidURL
        with httpx.Client(timeout=httpx.Timeout(900.0), trust_env=False) as client:
            with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    raw = resp.read().decode("utf-8", "replace")[:500]
                    raise LLMError(f"HTTP {resp.status_code}: {raw}")
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    ch = choices[0]
                    delta = ch.get("delta") or {}
                    if isinstance(delta.get("content"), str):
                        out.append(delta["content"])
                    elif isinstance(delta.get("reasoning"), str):
                        reason.append(delta["reasoning"])
                    elif isinstance(delta.get("reasoning_content"), str):
                        reason.append(delta["reasoning_content"])
                    if ch.get("finish_reason"):
                        finish = str(ch["finish_reason"])
        text = "".join(out)
        if not text.strip():
            text = "".join(reason)
        # 兼容 provider 把正文放在 delta.text 的变体
        return text, finish

    @staticmethod
    def _blocking_once(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Tuple[str, str]:
        import httpx

        body = dict(body)
        body["stream"] = False
        with httpx.Client(timeout=httpx.Timeout(900.0), trust_env=False) as client:
            resp = client.post(url, headers=headers, json=body)
            if resp.status_code != 200:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
        finish = ""
        try:
            finish = str(data["choices"][0].get("finish_reason") or "")
        except Exception:
            pass
        return extract_message_text(data), finish

    async def ask(
        self,
        prompt: str,
        model: Optional[str] = None,
        image: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **extra: Any,
    ) -> str:
        """异步入口：把阻塞 IO 丢到线程里，避免长时间占用事件循环。"""
        return await asyncio.to_thread(
            self._post_sync,
            prompt,
            model=model,
            image=image,
            api_key=api_key,
            base_url=base_url,
            **extra,
        )


class SmartMockLLM:
    """本地模拟模型，支持按关键词 / 模型名分发规则。"""

    def __init__(
        self,
        rules: Optional[Dict[str, Union[str, Callable[..., Any]]]] = None,
        model_rules: Optional[Dict[str, Union[str, Callable[..., Any]]]] = None,
        fallback_fn: Optional[Callable[..., str]] = None,
    ) -> None:
        self.rules: Dict[str, Union[str, Callable[..., Any]]] = dict(rules or {})
        self.model_rules: Dict[str, Union[str, Callable[..., Any]]] = dict(model_rules or {})
        self.fallback_fn = fallback_fn
        self.call_history: List[Dict[str, Any]] = []

    def add_rule(self, keyword: str, handler: Union[str, Callable[..., Any]]) -> None:
        self.rules[keyword] = handler

    def add_model_rule(self, model_name: str, handler: Union[str, Callable[..., Any]]) -> None:
        self.model_rules[model_name] = handler

    async def _invoke(
        self, handler: Union[str, Callable[..., Any]], prompt: str, model: Optional[str], image: Optional[str]
    ) -> str:
        if callable(handler):
            sig = inspect.signature(handler)
            n = len(
                [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            )
            res = handler(prompt, model, image) if n >= 3 else handler(prompt, model) if n == 2 else handler(prompt)
            if asyncio.iscoroutine(res):
                res = await res
            return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
        return str(handler)

    async def ask(
        self,
        prompt: str,
        model: Optional[str] = None,
        image: Optional[str] = None,
        **extra: Any,
    ) -> str:
        self.call_history.append({"prompt": prompt, "model": model, "image": image, "extra": extra})
        if model and model in self.model_rules:
            return await self._invoke(self.model_rules[model], prompt, model, image)
        for kw, handler in self.rules.items():
            if kw in prompt:
                return await self._invoke(handler, prompt, model, image)
        if self.fallback_fn is not None:
            return await self._invoke(self.fallback_fn, prompt, model, image)
        return f"[Processed]: {prompt}"


#: 兼容别名（旧代码按私有名引用过，保留以免破坏外部脚本）
_extract_message_text = extract_message_text
