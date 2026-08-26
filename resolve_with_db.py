"""
递归 Prompt 解析器 (Class 版本)

类型定义:
  Prompt = str | int | {"agent": str, "context": str | int | list[Prompt | str | int]}

核心语义：DAG 节点调度器
  - int 引用 → 只提取 response（前置任务结果），不掺杂 context
  - pending 条目 → 先递归执行，拿到 response 再返回
  - dict → 执行 API 获取此块输出
  - list → 递归每个元素后用 \n\n 拼接
"""

import json
from typing import Any
from opencode import run as opencode_run


class PromptResolver:
    def __init__(self, db, model: str = "", timeout: int = 600, use_cache: bool = False):
        self.db = db
        self.model = model
        self.timeout = timeout
        self.use_cache = use_cache
        self._cache = {}

    def resolve(self, prompt: Any) -> str:
        """
        解析给定的 Prompt 并返回最终的文本结果。
        输入必须是 dict（或可解析为 dict 的 JSON str），否则报错。
        """
        # 1. 归一化为 dict
        data = self._normalize(prompt)

        agent = data.get("agent", "")
        context = data.get("context", "")

        # 2. 解析依赖树
        resolved_context_text = self._resolve_element(context)

        if not resolved_context_text.strip():
            return ""

        # 3. 执行最终的主任务
        result = opencode_run(
            resolved_context_text, agent=agent, model=self.model, timeout=self.timeout
        )

        if result.get("success"):
            return self._to_text(result.get("output", ""))
        return ""

    def _normalize(self, prompt: Any) -> dict:
        """将输入强制转为 dict。"""
        if isinstance(prompt, str):
            try:
                data = json.loads(prompt)
                if not isinstance(data, dict):
                    raise ValueError("JSON 必须解析为字典结构。")
                return data
            except json.JSONDecodeError:
                raise ValueError("输入的 prompt 必须是合法的 JSON 字符串。")
        if isinstance(prompt, dict):
            return prompt
        raise TypeError("输入的 prompt 只允许是 JSON 字符串或字典。")

    def _resolve_int(self, pid: int) -> str:
        """
        解析数据库中的 ID：
        - done → 只返回 response
        - pending → 先递归执行，拿到 response 再返回
        """
        row = self.db.get(pid)
        if row is None:
            return ""

        if row["status"] == "done" and row["response"]:
            return row["response"]

        context = row.get("context", "")
        if not context:
            self.db.done(pid, "", {"source": "empty_context"})
            return ""

        try:
            ctx_val = json.loads(context)
        except (json.JSONDecodeError, TypeError):
            ctx_val = context

        resolved = self._resolve_element(ctx_val)

        result = opencode_run(
            resolved,
            agent=row.get("agent", ""),
            model=self.model or row.get("model", ""),
            timeout=self.timeout,
        )
        final_text = self._to_text(result.get("output", ""))

        if result.get("success"):
            self.db.done(pid, final_text, {"source": "opencode_run"})
        else:
            self.db.failed(pid, final_text or "opencode call failed")

        return final_text

    def _resolve_element(self, element: Any) -> str:
        """递归处理元素的通用引擎。"""
        if isinstance(element, int):
            return self._resolve_int(element)

        if isinstance(element, str):
            return element

        if isinstance(element, list):
            parts = [self._resolve_element(item) for item in element]
            return "\n\n".join(parts)

        if isinstance(element, dict):
            agent = element.get("agent", "")
            context = element.get("context", "")
            resolved = self._resolve_element(context)
            if not resolved.strip():
                return ""
            result = opencode_run(
                resolved, agent=agent, model=self.model, timeout=self.timeout
            )
            if result.get("success"):
                return self._to_text(result.get("output", ""))
            return ""

        return str(element)

    def _to_text(self, output: Any) -> str:
        if isinstance(output, dict):
            return json.dumps(output, indent=2, ensure_ascii=False)
        return str(output)
