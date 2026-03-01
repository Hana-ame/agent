"""
write - 将 LAST_RESPONSE.txt 的内容写入文件

用法：py utils.py write <相对路径> [force]
参数：
  <相对路径>  目标文件路径
  force       可选参数，强制写入，跳过多文件块检测

注意：需要先通过纯文本输出内容，由 Agent 自动保存到 LAST_RESPONSE.txt，
然后执行此命令写入文件。LAST_RESPONSE.txt 位于根目录。

如果检测到内容中包含多个 "===" 开头的行（可能为多文件块），
且未使用 force 参数，则会提示建议使用 write_multiple。
"""

import os

PREVIEW_LENGTH = 250

def run(ctx, args):
    # 参数检查：允许 1 个或 2 个参数
    if len(args) not in (1, 2):
        return "错误：write 需要 1 个参数：<相对路径>，或 2 个参数：<相对路径> force"
    
    path = ctx.validate_path(args[0])
    force = False
    if len(args) == 2:
        if args[1].lower() == "force":
            force = True
        else:
            return "错误：第二个参数必须是 'force'"

    last_response_path = os.path.join(ctx.root_path, "LAST_RESPONSE.txt")
    try:
        with open(last_response_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"错误：无法读取 LAST_RESPONSE.txt - {e}"

    # 移除首尾的 ``` 标记
    lines = content.splitlines()
    if lines and lines[0].startswith("```"):
        lines.pop(0)
    if lines and lines[-1].startswith("```"):
        lines.pop()
    striped_content = "\n".join(lines)

    # 检测多文件块：是否存在以 "===" 开头的行
    content_lines = striped_content.splitlines()
    has_multiple_blocks = any(line.strip().startswith("===") for line in content_lines)

    if has_multiple_blocks and not force:
        return (
            "检测到内容中包含 '===' 开头的行，可能为多文件块。\n"
            "建议使用 'py utils.py write_multiple' 来批量写入。\n"
            "如果仍要强制使用 write 写入单个文件，请添加 force 参数：\n"
            "py utils.py write <路径> force"
        )

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(striped_content)
        # 回显写入内容
        if len(striped_content) > 200:
            return f"{striped_content[0:PREVIEW_LENGTH]}...(中间省略)...{striped_content[-PREVIEW_LENGTH:]}\n\n写入至：{args[0]}"
        return f"{striped_content}\n\n写入至：{args[0]}"
    except Exception as e:
        return f"错误：无法写入文件 {args[0]} - {e}"