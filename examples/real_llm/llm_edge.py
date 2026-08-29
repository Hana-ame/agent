"""Real LLM Edge — processes via a real OpenAI-compatible endpoint.

Loaded from ``config.json`` through ``script: llm_edge.py:HttpLLMEdge``.
The edge owns its agent explicitly (``HttpLLMAgent``); nothing is injected
by the runner and no fallback default agent is used.
"""

import logging

from framework.edge import Edge
from framework.vertex import EdgeSignal
from framework.agents import HttpLLMAgent

logger = logging.getLogger("llm_edge")


class HttpLLMEdge(Edge):
    """Edge that delegates the prompt to a real LLM endpoint via HttpLLMAgent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = HttpLLMAgent(
            base_url=self.settings.get(
                "base_url", "https://opencode.ai/zen/v1"
            ),
            proxy=self.settings.get("https_proxy"),
        )

    async def execute(self, source_vertex, dest_vertex, agents, **kwargs):
        """Minimal read -> agent -> deliver flow against the LLM endpoint.

        Uses ``self.agent`` (owned here), never the injected ``agents``.
        """
        logger.info(
            "[HttpLLMEdge:%s] delegating to LLM endpoint (model=%s)...",
            self.id, self.model,
        )
        try:
            data = await source_vertex.fetch_data(channel=self.channel)
            if data is None:
                raise ValueError(
                    f"No data received from source vertex '{self.source_id}'."
                )

            result = await self.agent.process(
                data=data,
                prompt=self.prompt,
                model=self.model,
                settings=self.settings,
            )

            logger.info(
                "[HttpLLMEdge:%s] received response: %s...",
                self.id, repr(result)[:50],
            )

            await dest_vertex.receive_signal(
                self.id,
                EdgeSignal.COMPLETED,
                payload=result,
                channel=self.channel,
            )
            self.completed = True
            self.result = result
            return result

        except Exception as exc:
            self.error = str(exc)
            logger.error(
                "[HttpLLMEdge:%s] FAILED: %s", self.id, exc, exc_info=True
            )
            await dest_vertex.receive_signal(
                self.id, EdgeSignal.FAILED, payload=str(exc)
            )
            raise
