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
            return "错误：无法定位 utils 目录。"

        tools =[]
        for filename in os.listdir(utils_dir):
            if filename.endswith(".py") and not filename.startswith("__") and filename not in ("core.py",):
                tools.append(filename[:-3])
        
        tools.sort()
        
        lines = ["可用工具列表："]
        for tool in tools:
            try:
                # 动态加载模块以尝试提取第一行注释作为简介
                module = importlib.import_module(f"utils.{tool}")
                doc = module.__doc__
                desc = ""
                if doc:
                    for line in doc.strip().splitlines():
                        if line.strip():
                            # 如果注释已经以工具名开头，去掉重复的工具名
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
                return f"工具 '{tool_name}' 存在，但没有提供详细的帮助文档。"
        except ImportError:
            return f"错误：找不到工具 '{tool_name}' (utils/{tool_name}.py 不存在)。"
    else:
        return "错误：参数过多。用法：py utils.py help[工具名]"