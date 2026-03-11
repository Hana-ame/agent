# [START] TOOL-WRITE-SNIPPET
"""
write_snippet - 增强版写入工具，支持代码段操作（--snippet）

用法：
    py utils.py write_snippet <路径> <内容...>                     # 普通写入
    py utils.py write_snippet --snippet NAME <路径> [--content 内容|--file 文件]   # 写入/更新代码段

选项：
    --snippet NAME   操作代码段，指定名称
    --content 内容   直接提供内容（可使用 \\n 表示换行）
    --file 文件      从文件读取内容
"""

import os
import sys
import re
from utils import snippet

def run(ctx, args):
    # 手动解析参数
    snippet_mode = False
    snippet_name = None
    content = None
    content_file = None
    path = None
    content_parts = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--snippet':
            snippet_mode = True
            if i + 1 < len(args) and not args[i+1].startswith('--'):
                snippet_name = args[i+1]
                i += 2
            else:
                return "错误：--snippet 缺少名称参数"
        elif arg == '--content':
            if i + 1 < len(args):
                content = args[i+1]
                i += 2
            else:
                return "错误：--content 缺少内容参数"
        elif arg == '--file':
            if i + 1 < len(args):
                content_file = args[i+1]
                i += 2
            else:
                return "错误：--file 缺少文件名参数"
        else:
            if path is None:
                path = arg
                i += 1
            else:
                content_parts.append(arg)
                i += 1

    if path is None:
        return "错误：缺少路径参数"

    final_content = None
    if content is not None:
        final_content = content.replace('\\n', '\n')
    elif content_file is not None:
        try:
            with open(ctx.validate_path(content_file), 'r', encoding='utf-8') as f:
                final_content = f.read()
        except Exception as e:
            return f"错误：无法读取内容文件 {content_file} - {e}"
    elif content_parts:
        final_content = " ".join(content_parts)
    else:
        print("请输入内容（以 Ctrl+Z 或 Ctrl+D 结束）：", file=sys.stderr)
        final_content = sys.stdin.read()
        if not final_content:
            return "错误：未提供内容"

    full_path = ctx.validate_path(path)

    if snippet_mode:
        if snippet_name is None:
            return "错误：代码段模式需要提供名称（--snippet NAME）"
        snippet_args = ['set', path, snippet_name, '--content', final_content]
        return snippet.run(ctx, snippet_args)
    else:
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(final_content)
            return f"Success: 已写入 {path} (长度: {len(final_content)})"
        except Exception as e:
            return f"Error: 写入失败 - {str(e)}"
# [END] TOOL-WRITE-SNIPPET