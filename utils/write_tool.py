# [START] TOOL-WRITE-TOOL
# version: 0.0.1
# 描述：专用于在 utils/ 文件夹中创建或更新工具脚本

"""
write_tool - 在 utils 目录中写入工具代码

用法：py utils.py write_tool <tool_name> <内容...>

参数：
  <tool_name> 工具名或文件名，例如 new_tool 或 new_tool.py
  <内容...>   要写入的 Python 脚本内容（支持 \n 换行）

说明：
  - 无论在哪运行系统，本命令都会强制将新工具写入到当前的 utils 文件夹内。
"""

import os

def run(ctx, args):
    if len(args) < 2:
        return "=== write_tool ===\n错误：write_tool 需要至少 2 个参数：<工具名> <内容...>\n=== end of write_tool ==="

    tool_name = args[0]
    if not tool_name.endswith(".py"):
        tool_name += ".py"

    content = " ".join(args[1:])

    try:
        # 获取当前 write_tool.py 的目录，即绝对路径的 utils 文件夹
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(utils_dir, tool_name)

        # 将字面量的 \n 转换为真实换行符
        content = content.replace("\\n", "\n")

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        preview_len = 250
        if len(content) > preview_len * 2:
            preview = content[:preview_len] + f"\n...(中间省略 {len(content)-preview_len*2} 字符)...\n" + content[-preview_len:]
        else:
            preview = content

        return f"=== write_tool ===\n成功在 {utils_dir} 中写入了工具：{tool_name}\n内容预览:\n{preview}\n=== end of write_tool ==="
    except Exception as e:
        return f"=== write_tool ===\n错误：{str(e)}\n=== end of write_tool ==="

# [END] TOOL-WRITE-TOOL