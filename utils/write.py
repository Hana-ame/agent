"""
write - 将 LAST_RESPONSE.txt 的内容写入文件

用法：py utils.py write <相对路径>

参数：
  <相对路径>  目标文件路径

注意：需要先通过纯文本输出内容，由 Agent 自动保存到 LAST_RESPONSE.txt，
然后执行此命令写入文件。LAST_RESPONSE.txt 位于根目录。
"""

import os

def run(ctx, args):
    if len(args) != 1:
        return "错误：write 需要 1 个参数：<相对路径>"
    path = ctx.validate_path(args[0])
    last_response_path = os.path.join(ctx.root_path, "LAST_RESPONSE.txt")
    try:
        with open(last_response_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"错误：无法读取 LAST_RESPONSE.txt - {e}"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"成功：已将 LAST_RESPONSE.txt 的内容写入 {args[0]}"
    except Exception as e:
        return f"错误：无法写入文件 {args[0]} - {e}"
