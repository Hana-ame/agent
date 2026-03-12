# [START] TOOL-WRITE
# version: 002
# 描述：覆盖写入文件

"""
write - 覆盖写入文件

用法：py utils.py write <相对路径> <内容...>

参数：
  <相对路径>  相对于根目录的文件路径
  <内容...>   要写入的文本内容（多个参数自动用空格连接）

自动创建父目录。
"""

import os
from . import _file_utils

PREVIEW_LENGTH = _file_utils.PREVIEW_LENGTH

def run(ctx, args):
    if len(args) < 2:
        return "Error: write 需要 2 个参数：<路径> <内容...>"
    
    rel_path = args[0]
    content = " ".join(args[1:])
    
    try:
        full_path = ctx.validate_path(rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content.replace(r"\n","\n"))
        
        # 生成预览
        if len(content) > PREVIEW_LENGTH * 2:
            preview = content[:PREVIEW_LENGTH] + "...(中间省略)..." + content[-PREVIEW_LENGTH:]
        else:
            preview = content
        
        return f"{rel_path}中被写入了以下内容\n{preview}"
    except Exception as e:
        return f"=== {rel_path} ===\n错误：{str(e)}\n=== end of {rel_path} ==="

# [END] TOOL-WRITE