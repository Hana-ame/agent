"""ShellCmd Edge — Executes arbitrary shell commands and passes stdout as edge result.

Usage (graph config):
{
  "edges": [{
    "id": "e_ls",
    "source": "src",
    "destination": "sink",
    "channel": "text",
    "script": "shell_edge.py:ShellCmdEdge",
    "settings": {
      "command": "ls -la framework",
      "timeout": 10
    }
  }]
}
"""

import asyncio
import logging
import os

from framework.edge import Edge

logger = logging.getLogger("shell_edge")


class ShellCmdEdge(Edge):
    """Executes shell command and passes stdout as edge result downstream."""

    async def compute(self, data, agent, settings):
        """Execute the command specified in settings.command."""
        cmd = settings.get("command", "echo 'no command'")
        timeout = settings.get("timeout", 30)

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"TIMEOUT: command took longer than {timeout}s"

        output = stdout.decode("utf-8", "replace").strip()
        if stderr:
            err = stderr.decode("utf-8", "replace").strip()
            output = f"{output}\n[stderr]\n{err}"
        return output
