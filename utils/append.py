"""
append - 将指定内容追加到文件末尾

用法：py utils.py append <相对路径> <内容>

参数：
  <相对路径>  相对于根目录的文件路径
  <内容>      要追加的文本内容（多个参数自动用空格连接）

如果文件不存在，会自动创建。
自动处理换行：如果文件非空且不以换行结尾，则先添加换行再追加，否则直接追加。
"""

import os
from . import _file_utils

PREVIEW_LENGTH = _file_utils.PREVIEW_LENGTH

def run(ctx, args):
    if len(args) < 2:
        return "错误：append 需要 2 个参数：<相对路径> <内容>"
    
    rel_path = args[0]
    content = " ".join(args[1:])
    
    try:
        full_path = ctx.validate_path(rel_path)
        # 确保父目录存在
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # 处理换行逻辑
        with open(full_path, 'a+', encoding='utf-8') as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(f.tell() - 1, os.SEEK_SET)
                last_char = f.read(1)
                if last_char != '\n':
                    f.write('\n')
            f.write(content)
        
        # 生成预览
        if len(content) > PREVIEW_LENGTH * 2:
            preview = content[:PREVIEW_LENGTH] + "...(中间省略)..." + content[-PREVIEW_LENGTH:]
        else:
            preview = content
        
        return f"{rel_path}中被追加了以下内容\n{preview}"
    except Exception as e:
        return f"=== {rel_path} ===\n错误：{str(e)}\n=== end of {rel_path} ==="