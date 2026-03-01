"""
pause - 暂停 Agent 执行，等待人工干预

用法：py utils.py pause

在根目录创建 .pause 文件，Agent 检测到该文件后会暂停循环，
直到执行 resume 或手动删除 .pause 文件。
"""

import os

def run(ctx, args):
    pause_file = os.path.join(ctx.root_path, ".pause")
    try:
        with open(pause_file, "w") as f:
            f.write("paused")
        return ""
    except Exception as e:
        return f"错误：无法设置暂停标志 - {e}"
