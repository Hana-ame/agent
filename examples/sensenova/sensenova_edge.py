"""SenseNova Edge — processes via the free SenseNova endpoint, no proxy.

Loaded from ``config.json`` through ``script: sensenova_edge.py:SensenovaEdge``.
The edge owns its ``HttpLLMAgent`` in Python; the base URL / model come from
``settings``, the API key from ``SENSENOVA_API_KEY`` (no transport proxy
needed — SenseNova is directly reachable).
"""

import logging
import os

from framework.edge import Edge
from framework.vertex import EdgeSignal
from framework.agents import HttpLLMAgent

logger = logging.getLogger("sensenova_edge")


class SensenovaEdge(Edge):
    """Edge that delegates the prompt to the SenseNova chat-completions endpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = HttpLLMAgent(
            base_url=self.settings.get("base_url", "https://token.sensenova.cn/v1"),
            api_key=os.environ.get("SENSENOVA_API_KEY", "public"),
        )

    async def execute(self, source_vertex, dest_vertex, agents, **kwargs):
        """Minimal read -> agent -> deliver flow against SenseNova.

        Uses ``self.agent`` (owned here), never the injected ``agents``.
        """
        logger.info(
            "[SensenovaEdge:%s] delegating to SenseNova (model=%s)...",
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
                "[SensenovaEdge:%s] received response: %s...",
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
                "[SensenovaEdge:%s] FAILED: %s", self.id, exc, exc_info=True
            )
            await dest_vertex.receive_signal(
                self.id, EdgeSignal.FAILED, payload=str(exc)
            )
            raise
