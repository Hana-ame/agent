#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
help - 查看工具列表或获取特定工具的帮助信息

用法：
    py utils.py help               # 列出所有可用工具
    py utils.py help <tool_name>   # 查看特定工具的使用说明

示例：
    py utils.py help write
    py utils.py help write_multiple
"""

import os
import importlib

def run(ctx, args):
    if len(args) == 0:
        # 获取 utils 目录
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isdir(utils_dir):
            return f"=== help ===\n错误：无法定位 utils 目录。\n=== end of help ==="

        tools = []
        for filename in os.listdir(utils_dir):
            if filename.endswith(".py") and not filename.startswith("__") and filename not in ("core.py",):
                tools.append(filename[:-3])

        tools.sort()

        lines = ["可用工具列表："]
        for tool in tools:
            try:
                module = importlib.import_module(f"utils.{tool}")
                doc = module.__doc__
                desc = ""
                if doc:
                    for line in doc.strip().splitlines():
                        if line.strip():
                            if line.strip().startswith(f"{tool} -"):
                                desc = " - " + line.strip().split("-", 1)[1].strip()
                            else:
                                desc = " - " + line.strip()
                            break
                lines.append(f"  {tool:<18}{desc}")
            except Exception:
                lines.append(f"  {tool:<18}")

        lines.append("\n使用 'py utils.py help <工具名>' 查看特定工具的详细说明。")
        lines.append("示例：")
        lines.append("  py utils.py help write")
        lines.append("  py utils.py help write_multiple")
        return "\n".join(lines)

    elif len(args) == 1:
        tool_name = args[0]
        try:
            module_name = f"utils.{tool_name}"
            module = importlib.import_module(module_name)
            doc = module.__doc__
            if doc:
                return doc.strip()
            else:
                return f"=== help ===\n错误：工具 '{tool_name}' 存在，但没有提供详细的帮助文档。\n=== end of help ==="
        except ImportError:
            return f"=== help ===\n错误：找不到工具 '{tool_name}' (utils/{tool_name}.py 不存在)。\n=== end of help ==="
    else:
        return f"=== help ===\n错误：参数过多。用法：py utils.py help [工具名]\n=== end of help ==="