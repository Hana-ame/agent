"""OpenCode CLI Edge — processes via the local ``opencode`` CLI.

Loaded from ``config.json`` through ``script: zen_edge.py:OpenCodeEdge``.
The edge owns its agent explicitly (``OpenCodeAgentRunner``); nothing is
injected by the runner and no fallback default agent is used — every
behaviour is declared in the config.
"""

import logging

from framework.edge import Edge
from framework.vertex import EdgeSignal
from framework.agents.opencode_agent_runner import OpenCodeAgentRunner

logger = logging.getLogger("opencode_edge")


class OpenCodeEdge(Edge):
    """Edge that delegates the prompt to the local ``opencode`` CLI."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = OpenCodeAgentRunner()

    async def execute(self, source_vertex, dest_vertex, agents, **kwargs):
        """Minimal read -> agent -> deliver flow against the opencode CLI.

        Uses ``self.agent`` (owned here), never the injected ``agents``.
        """
        logger.info(
            "[OpenCodeEdge:%s] delegating to opencode CLI (model=%s)...",
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
                "[OpenCodeEdge:%s] received response: %s...",
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
                "[OpenCodeEdge:%s] FAILED: %s", self.id, exc, exc_info=True
            )
            await dest_vertex.receive_signal(
                self.id, EdgeSignal.FAILED, payload=str(exc)
            )
            raise
