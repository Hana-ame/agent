# [START] TOOL-LIST
# version: 003
# 描述：列出目录内容

"""
ls - 列出目录中的文件和子目录

用法：py utils.py ls [目录路径]

参数：
  [目录路径]   要列出的目录（默认：当前目录）

输出：
  成功时每行一个条目，目录后加 /
  失败时输出错误信息，格式为：
  === 路径 ===
  错误：具体错误信息
  === end of 路径 ===
"""

import os

def run(ctx, args):
    rel_path = args[0] if args else "."
    
    try:
        full_path = ctx.validate_path(rel_path)
        if not os.path.isdir(full_path):
            return f"=== {rel_path} ===\n错误：不是一个目录\n=== end of {rel_path} ==="
            
        items = os.listdir(full_path)
        if not items:
            return "(空目录)"
        
        result = []
        for item in items:
            item_path = os.path.join(full_path, item)
            suffix = "/" if os.path.isdir(item_path) else ""
            result.append(f"{item}{suffix}")
            
        return "\n".join(result)
    except Exception as e:
        return f"=== {rel_path} ===\n错误：{str(e)}\n=== end of {rel_path} ==="
# [END] TOOL-LIST