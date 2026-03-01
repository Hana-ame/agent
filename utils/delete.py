
# utils/delete.py

import os

def run(ctx, args):
    if len(args) != 1:
        return "错误：delete 需要 1 个参数：<相对路径>"
    path = ctx.validate_path(args[0])
    try:
        os.remove(path)
        return f"成功：已删除文件 {args[0]}"
    except Exception as e:
        return f"错误：无法删除文件 {args[0]} - {e}"