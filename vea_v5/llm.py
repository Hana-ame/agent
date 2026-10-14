"""VEA V5 模型推理接口模块 (LLM Backend).

提供极简、零外部强依赖的大模型调用协议：
1. `LLMProtocol`：任何实现 `async def ask(prompt: str) -> str` 或可调用对象的模型后端均可直接接入。
2. `OpenAICompatibleLLM`：原生支持任何 OpenAI 兼容接口（如 OpenAI、SenseNova、DeepSeek、Ollama、vLLM）。
3. `SmartMockLLM`：内置的可编程/启发式模拟大模型，使所有示例与单元测试在无外网或无 API Key 时也能零配置跑通。
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Callable, Dict, List, Optional, Protocol, Union, runtime_checkable


@runtime_checkable
class LLMProtocol(Protocol):
    """VEA V5 极简大模型协议：只需实现 `ask(prompt)` 即可。"""

    async def ask(self, prompt: str) -> str: ...


class OpenAICompatibleLLM:
    """标准 OpenAI 兼容 HTTP 客户端（支持 SenseNova / OpenAI / DeepSeek / Ollama）。"""

    def __init__(
        self,
        model: str = "sensenova-6.8-flash-lite",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("VEA_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.base_url = (
            base_url
            or os.environ.get("VEA_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1/chat/completions"
        )
        if not self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url.rstrip("/") + "/chat/completions"
        self.temperature = temperature
        self.timeout = timeout

    async def ask(self, prompt: str) -> str:
        """发送单轮提示词并返回模型生成的文本。"""
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        async with httpx.AsyncClient(headers=headers, timeout=self.timeout) as client:
            resp = await client.post(self.base_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()


class SmartMockLLM:
    """支持自定义规则路由与默认智能响应的本地模拟 LLM（用于离线测试与演示）。

    可以通过 `rules` 字典传入 `{"关键词": "返回结果或处理函数"}`，未命中规则时回退到默认处理函数。
    """

    def __init__(
        self,
        rules: Optional[Dict[str, Union[str, Callable[[str], Any]]]] = None,
        fallback_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.rules: Dict[str, Union[str, Callable[[str], Any]]] = dict(rules or {})
        self.fallback_fn = fallback_fn
        self.call_history: List[str] = []

    def add_rule(self, keyword: str, handler: Union[str, Callable[[str], Any]]) -> None:
        """注册一条关键词响应规则。"""
        self.rules[keyword] = handler

    async def ask(self, prompt: str) -> str:
        """按匹配到的最长关键词规则生成响应。"""
        self.call_history.append(prompt)
        # 按关键词长度降序匹配，优先匹配更具体的规则
        for keyword in sorted(self.rules.keys(), key=len, reverse=True):
            if keyword in prompt:
                handler = self.rules[keyword]
                if callable(handler):
                    res = handler(prompt)
                    if asyncio.iscoroutine(res):
                        res = await res
                    return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
                return str(handler)

        if self.fallback_fn is not None:
            return str(self.fallback_fn(prompt))

        # 默认回退：提取输入数据部分做摘要标记
        if "输入数据：" in prompt:
            instruction, data = prompt.split("输入数据：", 1)
            return f"[{instruction.strip()} 的处理结果] -> {data.strip()}"
        return f"[MockLLM 已处理]: {prompt[:120]}"
