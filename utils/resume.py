"""
resume - 恢复 Agent 执行（删除暂停标志）

用法：py utils.py resume

删除根目录的 .pause 文件，让 Agent 继续执行。
"""

import os

def run(ctx, args):
    pause_file = os.path.join(ctx.root_path, ".pause")
    try:
        if os.path.exists(pause_file):
            os.remove(pause_file)
            return "已删除暂停标志，Agent 将恢复执行。"
        else:
            return "没有暂停标志（.pause 文件不存在）。"
    except Exception as e:
        return f"=== resume ===\n错误：{str(e)}\n=== end of resume ==="