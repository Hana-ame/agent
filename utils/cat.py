# [START] TOOL-READ
# version: 003
# 描述：读取文件内容，按统一格式输出

"""
cat - 读取一个或多个文件的内容

用法：py utils.py cat <文件路径1> [<文件路径2> ...]

输出格式：
  === 文件路径 ===
  文件内容（或错误信息）
  === end of 文件路径 ===

多个文件之间用空行分隔。
"""

import os

def run(ctx, args):
    if not args:
        return "错误：cat 需要至少一个文件路径"
    
    results = []
    for rel_path in args:
        try:
            full_path = ctx.validate_path(rel_path)
            if not os.path.exists(full_path):
                content = f"错误：文件不存在"
            else:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            # 统一格式
            block = f"=== {rel_path} ===\n{content}\n=== end of {rel_path} ==="
            results.append(block)
        except Exception as e:
            block = f"=== {rel_path} ===\n错误：{str(e)}\n=== end of {rel_path} ==="
            results.append(block)
    
    return "\n\n".join(results)
# [END] TOOL-READ