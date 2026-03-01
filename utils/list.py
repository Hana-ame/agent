
# utils/list.py

import os

def run(ctx, args):
    if len(args) != 1:
        return "错误：list 需要 1 个参数：<相对路径>"
    path = ctx.validate_path(args[0])
    try:
        items = os.listdir(path)
        return "\n".join(items) if items else "(空目录)"
    except Exception as e:
        return f"错误：无法列出目录 {args[0]} - {e}"