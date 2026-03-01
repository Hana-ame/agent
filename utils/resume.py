"""
resume - 恢复被暂停的 Agent 执行

用法：py utils.py resume

删除根目录的 .pause 文件，使 Agent 继续执行。
"""

import os

def run(ctx, args):
    pause_file = os.path.join(ctx.root_path, ".pause")
    if os.path.exists(pause_file):
        try:
            os.remove(pause_file)
            return "已清除暂停标志。可以继续执行命令。"
        except Exception as e:
            return f"错误：无法清除暂停标志 - {e}"
    else:
        return "当前没有暂停标志。"
