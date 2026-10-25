"""Pi Agent Edge — delegates execution to the real Pi Agent CLI.

This module mirrors ``examples/real_llm/llm_edge.py`` but, instead of
constructing a raw HTTP request to an OpenAI-compatible endpoint, it
hands the payload to the agent injected by the runner — wired to
:class:`framework.agents.PiAgentRunner`, which shells out to the
installed ``pi`` CLI.
"""

import logging

from framework.edge import Edge
from framework.vertex import EdgeSignal

logger = logging.getLogger("pi_edge")


class PiEdge(Edge):
    """Edge that bypasses the default 5-stage pipeline and invokes the Pi CLI.

    ``execute`` skips guard / pre-process / retry / schema stages and runs
    a minimal read -> agent -> deliver flow. The agent supplied by the
    executor is a :class:`PiAgentRunner` (see ``examples/run.py``), which
    turns each call into a ``pi -p --model <model> --system-prompt <prompt>
    -- <data>`` subprocess.
    """

    async def execute(self, source_vertex, dest_vertex, agents, **kwargs):
        """Run a minimal read -> agent -> deliver flow against the Pi CLI."""
        logger.info(
            "[PiEdge:%s] Intercepted execution to delegate to Pi Agent CLI",
            self.id,
        )
        try:
            # 1 — Fetch data from source vertex
            data = await source_vertex.fetch_data(channel=self.channel)
            if data is None:
                raise ValueError(
                    f"No data received from source vertex '{self.source_id}'."
                )

            logger.info(
                "[PiEdge:%s] Delegating to Pi Agent (model=%s)...",
                self.id, self.model,
            )

            # 2 — Compute via the injected agent (PiAgentRunner at runtime)
            result = await agents.process(
                data=data,
                prompt=self.prompt,
                model=self.model,
                settings=self.settings,
            )

            logger.info(
                "[PiEdge:%s] Received response: %s...",
                self.id, repr(result)[:50],
            )

            # 3 — Deliver to destination vertex
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
                "[PiEdge:%s] FAILED: %s", self.id, exc, exc_info=True
            )
            await dest_vertex.receive_signal(
                self.id, EdgeSignal.FAILED, payload=str(exc)
            )
            raise
