# [START] TOOL-WRITE
# version: 0.0.3
# 描述：覆盖写入文件

"""
write - 覆盖写入文件

用法：py utils.py write <相对路径> <内容...>

参数：
  <相对路径>  相对于根目录的文件路径
  <内容...>   要写入的文本内容（多个参数自动用空格连接）

说明：
  - 自动创建父目录。
  - 可以使用 `\\n` 表示换行符（会被转换为真实换行）。
  - 内容较长时会自动截断预览。

输出格式：
  - 成功时：返回一行：
    {rel_path}中被写入了以下内容
    {内容预览}
  - 失败时：
    * 参数不足返回：
      === write ===
      错误：write 需要至少 2 个参数：<路径> <内容...>
      === end of write ===
    * 写入失败返回针对该文件的错误块：
      === {rel_path} ===
      错误：具体错误信息
      === end of {rel_path} ===
"""

import os
from . import _file_utils

def _handle_error(subcmd: str, msg: str) -> str:
    return f"=== {subcmd} ===\n错误：{msg}\n=== end of {subcmd} ==="

def run(ctx, args):
    if len(args) < 2:
        return _handle_error("write", "write 需要至少 2 个参数：<路径> <内容...>")

    rel_path = args[0]
    content = " ".join(args[1:])

    try:
        full_path = ctx.validate_path(rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # 将字面量 \n 转换为真实换行符
        content = content.replace("\\n", "\n")

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        preview_len = getattr(_file_utils, 'PREVIEW_LENGTH', 250)
        if len(content) > preview_len * 2:
            preview = content[:preview_len] + f"...(中间省略 {len(content)-preview_len*2} 字符)..." + content[-preview_len:]
        else:
            preview = content

        return f"{rel_path}中被写入了以下内容\n{preview}"
    except Exception as e:
        return _handle_error(rel_path, str(e))

# [END] TOOL-WRITE