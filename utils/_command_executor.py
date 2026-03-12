"""
_command_executor - 内部模块，用于执行从回复中提取的命令。
此模块不提供直接命令行调用，仅供其他工具内部使用。
"""

import sys
import os
import asyncio
import shlex
import subprocess
from . import _file_utils

class CommandExecutor:
    """负责执行从 LLM 回复中提取的命令，并收集输出"""

    @staticmethod
    async def execute_command(cmd: str) -> str:
        """
        执行单条命令，返回输出。
        命令格式如 'py utils.py ...'，会转换为绝对路径并设置工作目录。
        """
        try:
            parts = shlex.split(cmd)
            if parts and parts[0] in ("py", "python", "python3"):
                parts[0] = sys.executable
                if len(parts) > 1 and os.path.basename(parts[1]) == "utils.py":
                    parts[1] = os.path.join(_file_utils.UTILS_PATH, "utils.py")

            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        parts,
                        capture_output=True,
                        text=True,
                        cwd=_file_utils.ROOT_PATH,
                        timeout=_file_utils.COMMAND_TIMEOUT
                    )
                ),
                timeout=_file_utils.COMMAND_TIMEOUT + 5
            )

            cmd_output = result.stdout + result.stderr
            if result.returncode != 0:
                cmd_output = f"命令执行失败 (返回码 {result.returncode}):\n{cmd_output}"
            else:
                cmd_output = cmd_output.strip()

        except subprocess.TimeoutExpired:
            cmd_output = f"命令执行超时 (超过 {_file_utils.COMMAND_TIMEOUT} 秒)"
        except asyncio.TimeoutError:
            cmd_output = f"命令执行超时 (整体等待超时)"
        except Exception as e:
            cmd_output = f"执行命令时出错: {str(e)}"

        hint = f"command {cmd} has been run, the result returned:"
        if cmd_output:
            return f"{hint}\n{cmd_output}"
        else:
            return hint + " (no output)"

    @classmethod
    async def execute_all(cls, commands: list[str]) -> str:
        """并发执行多条命令，合并输出"""
        if not commands:
            return ""
        tasks = [cls.execute_command(cmd) for cmd in commands]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        formatted = []
        for i, out in enumerate(outputs):
            if isinstance(out, Exception):
                formatted.append(f"命令 {commands[i]} 执行异常: {str(out)}")
            else:
                formatted.append(out)
        return "\n\n".join(formatted)