"""Launch the ``opencode`` CLI (headless ``opencode run``) to process a message.

Subprocess agent, the opencode counterpart of :class:`PiAgentRunner`: instead
of talking HTTP to an endpoint, it shells out to the local ``opencode`` CLI
and returns the assistant's reply. An edge loads it via ``script: 文件:类``
and owns it directly — nothing is injected by the runner.
"""

import json
import logging
from typing import Any, Dict, Optional

from .base_agent import BaseAgent

logger = logging.getLogger("vertex_edge_agent.agents")


class OpenCodeAgentRunner(BaseAgent):
    """Run ``opencode run <message>`` and return the reply."""

    async def process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> Any:
        if self.mock:
            return await self._mock_process(data, prompt, model, settings)
        import asyncio

        settings = settings or {}

        if isinstance(data, (dict, list)):
            message = json.dumps(data, ensure_ascii=False)
        else:
            message = str(data)

        cmd = ["opencode", "run"]
        if model and model != "default":
            cmd.extend(["--model", model])
        if settings.get("format") == "json":
            cmd.extend(["--format", "json"])
        if settings.get("agent"):
            cmd.extend(["--agent", str(settings["agent"])])

        # ``opencode run`` has no --system-prompt flag; fold the prompt in.
        if prompt:
            message = f"{prompt}\n\n{message}"
        cmd.append(message)

        logger.debug("[OpenCodeAgentRunner] executing: %s", " ".join(cmd))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error(
                "[OpenCodeAgentRunner] 'opencode' command not found in PATH. "
                "Install it (e.g. `curl -fsSL https://opencode.ai/install | bash`) first."
            )
            raise

        # ``settings["timeout"]`` bounds the CLI call; when the enclosing task
        # is cancelled (executor/edge timeout) we kill the child so no orphaned
        # ``opencode`` process is left behind.
        timeout = settings.get("timeout")
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise RuntimeError(
                f"opencode run timed out after {timeout}s, killed."
            ) from None
        except asyncio.CancelledError:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise

        if proc.returncode != 0:
            err = stderr.decode("utf-8", "replace").strip()
            logger.error(
                "[OpenCodeAgentRunner] opencode run failed (code %s): %s",
                proc.returncode, err,
            )
            raise RuntimeError(f"opencode run failed ({proc.returncode}): {err}")

        output = stdout.decode("utf-8", "replace").strip()

        if settings.get("format") == "json":
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                logger.warning(
                    "[OpenCodeAgentRunner] --format json returned unparsable output; "
                    "falling back to raw string."
                )
        return output
