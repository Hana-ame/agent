"""
list - 列出目录内容

用法：py utils.py list <相对路径>

参数：
  <相对路径>  要列出的目录路径（相对于根目录）

返回该目录下的所有文件和文件夹名称，每行一个。
"""

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
