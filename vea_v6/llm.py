"""VEA V6 模型推理与多模态处理接口模块 (LLM Backend).

支持 V6 边处理方式中的：
1. 纯字符串 `prompt: str`
2. 指定模型与处理提示词的字典 `{"模型": "model_name", "处理prompt": "prompt_str"}`
   （兼容 `{"model": "model_name", "prompt": "prompt_str"}`）
3. 携带 `image?: path|url` 的多模态信息条目推理
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from typing import Any, Callable, Dict, List, Optional, Protocol, Union, runtime_checkable


@runtime_checkable
class LLMProtocol(Protocol):
    """VEA V6 大模型调用协议。"""

    async def ask(
        self,
        prompt: str,
        model: Optional[str] = None,
        image: Optional[str] = None,
    ) -> str: ...


class OpenAICompatibleLLM:
    """支持动态模型切换与多模态图片输入的 OpenAI 兼容客户端。"""

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

    async def ask(
        self,
        prompt: str,
        model: Optional[str] = None,
        image: Optional[str] = None,
    ) -> str:
        """发送提示词（及可选图片）到指定模型并返回生成的文本。"""
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        target_model = model or self.model
        if image:
            user_content: Any = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image}},
            ]
        else:
            user_content = prompt

        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": self.temperature,
        }
        async with httpx.AsyncClient(headers=headers, timeout=self.timeout) as client:
            resp = await client.post(self.base_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()


class SmartMockLLM:
    """支持关键词规则、模型路由规则与多模态图片感知的本地模拟大模型（零外部依赖）。"""

    def __init__(
        self,
        rules: Optional[Dict[str, Union[str, Callable[..., Any]]]] = None,
        model_rules: Optional[Dict[str, Union[str, Callable[..., Any]]]] = None,
        fallback_fn: Optional[Callable[..., str]] = None,
    ) -> None:
        self.rules: Dict[str, Union[str, Callable[..., Any]]] = dict(rules or {})
        self.model_rules: Dict[str, Union[str, Callable[..., Any]]] = dict(model_rules or {})
        self.fallback_fn = fallback_fn
        self.call_history: List[Dict[str, Optional[str]]] = []

    def add_rule(self, keyword: str, handler: Union[str, Callable[..., Any]]) -> None:
        """注册一条按提示词关键词触发的响应规则。"""
        self.rules[keyword] = handler

    def add_model_rule(self, model_name: str, handler: Union[str, Callable[..., Any]]) -> None:
        """注册一条按模型名称触发的响应规则。"""
        self.model_rules[model_name] = handler

    async def _invoke_handler(
        self,
        handler: Union[str, Callable[..., Any]],
        prompt: str,
        model: Optional[str],
        image: Optional[str],
    ) -> str:
        if callable(handler):
            sig = inspect.signature(handler)
            params = [
                p
                for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if len(params) >= 3:
                res = handler(prompt, model, image)
            elif len(params) == 2:
                res = handler(prompt, model)
            else:
                res = handler(prompt)
            if asyncio.iscoroutine(res):
                res = await res
            return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
        return str(handler)

    async def ask(
        self,
        prompt: str,
        model: Optional[str] = None,
        image: Optional[str] = None,
    ) -> str:
        """根据关键词规则或模型规则生成响应。"""
        self.call_history.append({"prompt": prompt, "model": model, "image": image})

        # 1. 按关键词长度降序匹配 rules
        for keyword in sorted(self.rules.keys(), key=len, reverse=True):
            if keyword in prompt:
                return await self._invoke_handler(self.rules[keyword], prompt, model, image)

        # 2. 按模型名称匹配 model_rules
        if model and model in self.model_rules:
            return await self._invoke_handler(self.model_rules[model], prompt, model, image)

        # 3. 自定义回退函数
        if self.fallback_fn is not None:
            return await self._invoke_handler(self.fallback_fn, prompt, model, image)

        # 4. 默认智能回退：保留模型、图片与输入摘要信息
        model_prefix = f"[{model}] " if model and model != "default" else ""
        img_suffix = f" (image={image})" if image else ""
        if "输入数据：" in prompt:
            instruction, data = prompt.split("输入数据：", 1)
            return f"{model_prefix}[{instruction.strip()}]{img_suffix} -> {data.strip()}"
        return f"{model_prefix}[Processed]{img_suffix}: {prompt[:160]}"
