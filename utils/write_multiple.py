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
        response = "\n".join(lines)
        return f"错误：{os.path.basename(response_file)} 格式错误（格式：=== 路径 === ... ===\n{os.path.basename(response_file)}文件预览:\n{response[:100]}...(中间省略)...{"\n".join(lines)[-100:]}"

    results = []
    for rel_path, file_content in files:
        # """保存响应内容到文件，移除首尾的```标记"""
        lines = file_content.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[0].startswith("```"):
            lines.pop(0)
        if lines and lines[-1].startswith("```"):
            lines.pop()
        striped_content = "\n".join(lines)
        try:
            full_path = ctx.validate_path(rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(striped_content)
            # 修改返回：去掉"成功"，回显内容，用 === 包裹
            
            result_str = f"{rel_path}中被写入了以下内容\n{striped_content}\n\n"
            if (len(striped_content) > 200):
                result_str = f"{rel_path}中被写入了以下内容\n{striped_content[0:100]}...(中间省略)...{striped_content[-100:]}\n\n"
            results.append(result_str)
        except Exception as e:
            results.append(f"错误：{rel_path} - {e}")

    return "\n\n".join(results)  # 用两个换行分隔每个文件块
 