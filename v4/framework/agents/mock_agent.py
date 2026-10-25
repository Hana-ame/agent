"""Backwards-compatible MockAgent wrapper around HttpLLMAgent(mock=True)."""
from typing import Optional, Callable
from .http_llm_agent import HttpLLMAgent


class MockAgent(HttpLLMAgent):
    """Backwards-compatibility mock agent.

    Delegates to HttpLLMAgent with native mock=True mode and optional response_fn / mock_handler.
    """

    def __init__(self, response_fn: Optional[Callable] = None, **kwargs):
        super().__init__(mock=True, mock_handler=response_fn, **kwargs)
