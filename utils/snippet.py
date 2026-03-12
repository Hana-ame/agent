"""
snippet - 管理文件中的代码段（以 # [START] name 或 // [START] name 标记）

用法：
    py utils.py snippet list <file>               列出文件中的所有代码段
    py utils.py snippet get <file> <name>         获取指定代码段的内容
    py utils.py snippet set <file> <name> [--content CONTENT] [--file CONTENT_FILE]  设置代码段内容
    py utils.py snippet delete <file> <name>      删除指定代码段

输出格式：
    成功时：返回普通文本（如成功信息或内容）。
    失败时：
      === snippet <子命令> ===
      错误：具体错误信息
      === end of snippet <子命令> ===
"""

import re
import os
import sys
import argparse
from pathlib import Path
from utils import core

# 匹配代码段，支持 # 和 // 作为注释符
SNIPPET_PATTERN = re.compile(
    r'^[ \t]*(?:#|//)[ \t]*\[START\][ \t]+([\w-]+)[ \t]*$\n?(.*?)^[ \t]*(?:#|//)[ \t]*\[END\][ \t]+\1[ \t]*$',
    re.DOTALL | re.MULTILINE
)

def find_snippets(content):
    """返回字典 {name: (start_idx, end_idx, start_line, end_line)}"""
    snippets = {}
    lines = content.splitlines(keepends=True)
    for match in SNIPPET_PATTERN.finditer(content):
        name = match.group(1)
        start_idx = match.start()
        end_idx = match.end()
        start_line = content.count('\n', 0, start_idx) + 1
        end_line = content.count('\n', 0, end_idx) + 1
        snippets[name] = (start_idx, end_idx, start_line, end_line)
    return snippets

def get_snippet_content(content, name):
    matches = list(SNIPPET_PATTERN.finditer(content))
    for match in matches:
        if match.group(1) == name:
            return match.group(2)
    return None

def replace_snippet(content, name, new_content):
    # 构建动态正则，匹配指定名称的代码段，支持 # 和 //
    pattern = re.compile(
        r'^[ \t]*(?:#|//)[ \t]*\[START\][ \t]+' + re.escape(name) + r'[ \t]*$\n?(.*?)^[ \t]*(?:#|//)[ \t]*\[END\][ \t]+' + re.escape(name) + r'[ \t]*$',
        re.DOTALL | re.MULTILINE
    )
    # 使用与匹配相同的注释符，但替换时固定用 # 还是保持原样？我们决定统一使用 # 作为新标记，因为简单。
    # 但为了保留原文件的注释风格，最好检测原文件使用的注释符，但实现复杂。此处简化：始终使用 #。
    new_block = f"# [START] {name}\n{new_content}\n# [END] {name}"
    if pattern.search(content):
        return pattern.sub(new_block, content, count=1)
    else:
        if not content.endswith('\n'):
            content += '\n'
        return content + new_block + '\n'

def delete_snippet(content, name):
    pattern = re.compile(
        r'^[ \t]*(?:#|//)[ \t]*\[START\][ \t]+' + re.escape(name) + r'[ \t]*$\n?(.*?)^[ \t]*(?:#|//)[ \t]*\[END\][ \t]+' + re.escape(name) + r'[ \t]*$\n?',
        re.DOTALL | re.MULTILINE
    )
    return pattern.sub('', content)

def _handle_error(subcmd: str, msg: str) -> str:
    return f"=== snippet {subcmd} ===\n错误：{msg}\n=== end of snippet {subcmd} ==="

def run(ctx: core.Context, args):
    parser = argparse.ArgumentParser(prog="py utils.py snippet", description="管理代码段", add_help=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_list = subparsers.add_parser("list", add_help=False)
    parser_list.add_argument("file", help="目标文件路径")

    parser_get = subparsers.add_parser("get", add_help=False)
    parser_get.add_argument("file", help="目标文件路径")
    parser_get.add_argument("name", help="代码段名称")

    parser_set = subparsers.add_parser("set", add_help=False)
    parser_set.add_argument("file", help="目标文件路径")
    parser_set.add_argument("name", help="代码段名称")
    group = parser_set.add_mutually_exclusive_group()
    group.add_argument("--content", help="直接提供内容（可使用 \\n 表示换行）")
    group.add_argument("--file", dest="content_file", help="从文件读取内容")

    parser_delete = subparsers.add_parser("delete", add_help=False)
    parser_delete.add_argument("file", help="目标文件路径")
    parser_delete.add_argument("name", help="代码段名称")

    try:
        parsed = parser.parse_args(args)
    except SystemExit:
        return "错误：参数解析失败"

    file_path = ctx.validate_path(parsed.file)
    subcmd = parsed.command

    if subcmd != "set" and not os.path.exists(file_path):
        return _handle_error(subcmd, f"文件 {parsed.file} 不存在")

    if subcmd == "list":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return _handle_error(subcmd, f"无法读取文件 {parsed.file} - {e}")

        snippets = find_snippets(content)
        if snippets:
            lines = [f"文件 {parsed.file} 中的代码段："]
            for name, (_, _, start, end) in snippets.items():
                lines.append(f"  {name} (行 {start}-{end})")
            return "\n".join(lines)
        else:
            return f"文件 {parsed.file} 中没有找到代码段。"

    elif subcmd == "get":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return _handle_error(subcmd, f"无法读取文件 {parsed.file} - {e}")

        snippet_content = get_snippet_content(content, parsed.name)
        if snippet_content is None:
            return _handle_error(subcmd, f"未找到代码段 '{parsed.name}'")
        return snippet_content

    elif subcmd == "set":
        if parsed.content:
            new_content = parsed.content.replace('\\n', '\n')
        elif parsed.content_file:
            try:
                content_file_path = ctx.validate_path(parsed.content_file)
                with open(content_file_path, 'r', encoding='utf-8') as f:
                    new_content = f.read()
            except Exception as e:
                return _handle_error(subcmd, f"无法读取内容文件 {parsed.content_file} - {e}")
        else:
            return _handle_error(subcmd, "必须提供 --content 或 --file 参数")

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = ""

        new_file_content = replace_snippet(content, parsed.name, new_content)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_file_content)
        return f"成功：代码段 '{parsed.name}' 已写入 {parsed.file}"

    elif subcmd == "delete":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return _handle_error(subcmd, f"无法读取文件 {parsed.file} - {e}")

        new_content = delete_snippet(content, parsed.name)
        if new_content == content:
            return _handle_error(subcmd, f"未找到代码段 '{parsed.name}'")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return f"成功：代码段 '{parsed.name}' 已从 {parsed.file} 删除"

    else:
        return _handle_error("unknown", f"未知的子命令: {subcmd}")