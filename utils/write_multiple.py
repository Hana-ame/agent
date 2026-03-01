"""
write_multiple - 批量写入多个文件（从指定的响应文件解析）

用法：
    py utils.py write_multiple              # 默认从 LAST_RESPONSE.txt 读取
    py utils.py write_multiple last          # 从 LAST_RESPONSE.txt 读取
    py utils.py write_multiple this          # 从 THIS_RESPONSE.txt 读取

响应文件（LAST_RESPONSE.txt 或 THIS_RESPONSE.txt）中需包含按以下格式组织的多个文件块：

    === 相对路径1 ===
    文件内容（可包含多行，保留原格式）
    ===
    === 相对路径2 ===
    文件内容2
    ===

每个块以 "=== 相对路径 ===" 开始，以单独一行的 "===" 结束。
工具会自动创建父目录。
"""

import os
import re

def run(ctx, args):
    # 解析参数，决定使用哪个响应文件
    if args and args[0] in ('this', 'last'):
        which = args[0]
    else:
        which = 'last'  # 默认 last
    
    if which == 'this':
        response_file = os.path.join(ctx.root_path, "THIS_RESPONSE.txt")
    else:  # 'last'
        response_file = os.path.join(ctx.root_path, "LAST_RESPONSE.txt")
    
    try:
        with open(response_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return f"错误：无法读取 {os.path.basename(response_file)} - {e}"

    files = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')
        start_match = re.match(r'^=== (.+) ===$', line)
        if start_match:
            rel_path = start_match.group(1).strip()
            i += 1
            content_lines = []
            while i < len(lines):
                current_line = lines[i].rstrip('\n')
                if current_line == "===":
                    break
                content_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].rstrip('\n') == "===":
                i += 1
            file_content = ''.join(content_lines)
            files.append((rel_path, file_content))
        else:
            i += 1

    if not files:
        return f"错误：{os.path.basename(response_file)} 中未找到任何有效文件块（格式：=== 路径 === ... ===\n{os.path.basename(response_file)}:\n{"\n".join(lines)}"

    results = []
    for rel_path, file_content in files:
        try:
            full_path = ctx.validate_path(rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            # 修改返回：去掉"成功"，回显内容，用 === 包裹
            result_str = f"{rel_path}中被写入了以下内容\n{file_content.rstrip()}\n\n"
            results.append(result_str)
        except Exception as e:
            results.append(f"错误：{rel_path} - {e}")

    return "\n\n".join(results)  # 用两个换行分隔每个文件块
