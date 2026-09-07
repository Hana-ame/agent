"""ShellCmd Edge — 执行任意 shell 命令并将输出作为 edge result。

用法（graph config）：
{
  "edges": [{
    "id": "e_ls",
    "source": "src",
    "destination": "sink",
    "channel": "text",
    "script": "shell_edge.py:ShellCmdEdge",
    "settings": {
      "command": "ls -la /mnt/d/WorkPlace/vertex_edge_agent/framework",
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
    """执行 shell 命令，stdout 作为 edge result 传递给下游 vertex。"""

    async def compute(self, data, agent, settings):
        """执行 settings.command 指定的 shell 命令。"""
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
