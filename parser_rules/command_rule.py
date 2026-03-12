# [START] COMMAND-RULE
# version: 0.0.1
# 上下文：拦截按行分割的单行终端指令。

import sys
import asyncio
from .base import BaseRule

class CommandRule(BaseRule):
    def __init__(self, root_path: str, utils_py_path: str):
        self.root_path = root_path
        self.utils_py_path = utils_py_path

    # [START] COMMAND-RULE-MATCH
    # version: 0.0.1
    # 上下文：逐行扫描大模型输出的文本。
    # 输入参数：text (str)
    # 输出参数：执行结果或标准错误 (str)
    async def match_and_execute(self, text: str) -> str:
        results =[]
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("py utils.py"):
                parts = line.split(" ", 2)
                args = parts[2].split() if len(parts) > 2 else []
                
                cmd =[sys.executable, self.utils_py_path] + args
                
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=self.root_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                
                out_str = stdout.decode('utf-8', errors='replace').strip()
                err_str = stderr.decode('utf-8', errors='replace').strip()
                
                res = f"执行命令 [{line}]:\n"
                if out_str: res += f"{out_str}\n"
                if err_str: res += f"错误:\n{err_str}\n"
                if not out_str and not err_str: res += "执行成功 (无输出)\n"
                
                results.append(res)
                
        return "\n".join(results)
    # [END] COMMAND-RULE-MATCH
# [END] COMMAND-RULE