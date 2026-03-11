"""
snippet - 管理文件中的代码段（以 # [START] name 和 # [END] name 标记）

用法：
    py utils.py snippet list <file>               列出文件中的所有代码段
    py utils.py snippet get <file> <name>         获取指定代码段的内容
    py utils.py snippet set <file> <name> [--content CONTENT] [--file CONTENT_FILE]  设置代码段内容
    py utils.py snippet delete <file> <name>      删除指定代码段
"""

import re
import os
import sys
import argparse
from pathlib import Path
from utils import core

SNIPPET_PATTERN = re.compile(
    r'^[ \\t]*#[ \\t]*\\[START\\][ \\t]+([\\w-]+)[ \\t]*$(.*?)^[ \\t]*#[ \\t]*\\[END\\][ \\t]+\\1[ \\t]*$',
    re.DOTALL | re.MULTILINE
)

def find_snippets(content):
    """返回字典 {name: (start_index, end_index, start_line, end_line)}"""
    snippets = {}
    lines = content.splitlines(keepends=True)
    for match in SNIPPET_PATTERN.finditer(content):
        name = match.group(1)
        start_idx = match.start()
        end_idx = match.end()
        start_line = content.count('\\n', 0, start_idx) + 1
        end_line = content.count('\\n', 0, end_idx) + 1
        snippets[name] = (start_idx, end_idx, start_line, end_line)
    return snippets

def get_snippet_content(content, name):
    matches = list(SNIPPET_PATTERN.finditer(content))
    for match in matches:
        if match.group(1) == name:
            return match.group(2)
    return None

def replace_snippet(content, name, new_content):
    pattern = re.compile(
        r'^[ \\t]*#[ \\t]*\\[START\\][ \\t]+' + re.escape(name) + r'[ \\t]*$(.*?)^[ \\t]*#[ \\t]*\\[END\\][ \\t]+' + re.escape(name) + r'[ \\t]*$',
        re.DOTALL | re.MULTILINE
    )
    new_block = f"# [START] {name}\\n{new_content}\\n# [END] {name}"
    if pattern.search(content):
        return pattern.sub(new_block, content, count=1)
    else:
        if not content.endswith('\\n'):
            content += '\\n'
        return content + new_block + '\\n'

def delete_snippet(content, name):
    pattern = re.compile(
        r'^[ \\t]*#[ \\t]*\\[START\\][ \\t]+' + re.escape(name) + r'[ \\t]*$(.*?)^[ \\t]*#[ \\t]*\\[END\\][ \\t]+' + re.escape(name) + r'[ \\t]*$\\n?',
        re.DOTALL | re.MULTILINE
    )
    return pattern.sub('', content)

def run(ctx: core.Context, args):
    parser = argparse.ArgumentParser(prog="py utils.py snippet", description="管理代码段")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_list = subparsers.add_parser("list", help="列出文件中的代码段")
    parser_list.add_argument("file", help="目标文件路径")

    parser_get = subparsers.add_parser("get", help="获取指定代码段的内容")
    parser_get.add_argument("file", help="目标文件路径")
    parser_get.add_argument("name", help="代码段名称")

    parser_set = subparsers.add_parser("set", help="设置代码段内容（创建或更新）")
    parser_set.add_argument("file", help="目标文件路径")
    parser_set.add_argument("name", help="代码段名称")
    group = parser_set.add_mutually_exclusive_group()
    group.add_argument("--content", help="直接提供内容（可使用 \\n 表示换行）")
    group.add_argument("--file", dest="content_file", help="从文件读取内容")

    parser_delete = subparsers.add_parser("delete", help="删除指定代码段")
    parser_delete.add_argument("file", help="目标文件路径")
    parser_delete.add_argument("name", help="代码段名称")

    try:
        parsed = parser.parse_args(args)
    except SystemExit:
        return 1

    file_path = ctx.validate_path(parsed.file)
    if not os.path.exists(file_path) and parsed.command != "set":
        print(f"错误：文件 {parsed.file} 不存在")
        return 1

    if parsed.command == "list":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"错误：无法读取文件 {parsed.file} - {e}")
            return 1
        snippets = find_snippets(content)
        if snippets:
            print(f"文件 {parsed.file} 中的代码段：")
            for name, (_, _, start, end) in snippets.items():
                print(f"  {name} (行 {start}-{end})")
        else:
            print(f"文件 {parsed.file} 中没有找到代码段。")
        return 0

    elif parsed.command == "get":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"错误：无法读取文件 {parsed.file} - {e}")
            return 1
        snippet_content = get_snippet_content(content, parsed.name)
        if snippet_content is None:
            print(f"错误：未找到代码段 '{parsed.name}'")
            return 1
        print(snippet_content, end='')
        return 0

    elif parsed.command == "set":
        if parsed.content:
            new_content = parsed.content.replace('\\n', '\n')
        elif parsed.content_file:
            try:
                with open(ctx.validate_path(parsed.content_file), 'r', encoding='utf-8') as f:
                    new_content = f.read()
            except Exception as e:
                print(f"错误：无法读取内容文件 {parsed.content_file} - {e}")
                return 1
        else:
            print("请输入代码段内容（以 Ctrl+Z 或 Ctrl+D 结束）：", file=sys.stderr)
            new_content = sys.stdin.read()
            if not new_content:
                print("错误：未提供内容")
                return 1

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = ""

        new_file_content = replace_snippet(content, parsed.name, new_content)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_file_content)
        print(f"成功：代码段 '{parsed.name}' 已写入 {parsed.file}")
        return 0

    elif parsed.command == "delete":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"错误：无法读取文件 {parsed.file} - {e}")
            return 1
        new_content = delete_snippet(content, parsed.name)
        if new_content == content:
            print(f"错误：未找到代码段 '{parsed.name}'")
            return 1
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"成功：代码段 '{parsed.name}' 已从 {parsed.file} 删除")
        return 0

    else:
        parser.print_help()
        return 1
