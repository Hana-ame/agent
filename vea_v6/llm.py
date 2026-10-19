"""兼容入口：`vea_v6.llm` → `vea_v6.parts.llm`。

拆分 `vea_v6` 为 `parts/` 之后，旧代码里的 `from vea_v6.llm import ...` 仍然有效。
不建议在新代码里用它；生成物只认 `parts/` 下的真实文件。
"""

from vea_v6.parts.llm import *  # noqa: F401,F403
from vea_v6.parts.llm import LLMError, LLMProtocol, OpenAICompatibleLLM, SmartMockLLM, extract_message_text

_extract_message_text = extract_message_text

__all__ = [
    "LLMProtocol",
    "LLMError",
    "extract_message_text",
    "OpenAICompatibleLLM",
    "SmartMockLLM",
]
