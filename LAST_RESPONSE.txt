#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils.py - 工具调用入口

支持两种调用方式：
  1. 命令行模式：py utils.py <工具名称> [参数...]
  2. 批量模式：py utils.py （无参数）—— 此时会读取 LAST_PROMPT.txt 中的命令并批量执行。
      LAST_PROMPT.txt 应包含多行，每行以 "py utils.py " 开头，后跟工具名称和参数。

工具列表（位于 utils/ 目录下）：
    read <相对路径>
    write <相对路径>
    append <相对路径> <内容>
    replace <相对路径> <旧文本> <新文本>
    list <相对路径>
    delete <相对路径> [多个路径...]
    move <源路径> <目标路径>
    pause
    resume
    write_multiple
    git <git命令...>
    ... 可扩展

使用 'py utils.py help' 查看所有工具。
"""

import sys
import importlib
import os
import shlex

def show_help(tool_name=None):
    """显示帮助信息"""
    utils_dir = os.path.join(os.path.dirname(__file__), "utils")
    if tool_name is None:
        # 显示所有工具列表
        print("可用工具：")
        if os.path.isdir(utils_dir):
            tools = []
            for f in os.listdir(utils_dir):
                if f.endswith(".py") and f not in ("__init__.py", "core.py"):
                    tools.append(f[:-3])
            tools.sort()
            for t in tools:
                # 尝试导入模块，获取文档字符串第一行
                try:
                    module = importlib.import_module(f"utils.{t}")
                    doc = module.__doc__.strip().split('\n')[0] if module.__doc__ else "无说明"
                except:
                    doc = "无法加载"
                print(f"  {t:15} - {doc}")
        print("\n使用 'py utils.py help <工具名>' 查看具体工具的帮助。")
    else:
        # 显示特定工具帮助
        try:
            module = importlib.import_module(f"utils.{tool_name}")
            if module.__doc__:
                print(module.__doc__.strip())
            else:
                print(f"工具 '{tool_name}' 没有提供帮助文档。")
        except ImportError:
            print(f"错误：未知的工具名称 '{tool_name}'")

def main():
    # 无参数模式：从 LAST_PROMPT.txt 读取命令批量执行
    if len(sys.argv) == 1:
        prompt_file = "LAST_PROMPT.txt"
        if not os.path.isfile(prompt_file):
            print(f"错误：找不到 {prompt_file}，请确保文件存在。")
            sys.exit(1)

        with open(prompt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 初始化上下文
        from utils import core
        ctx = core.Context(core.load_root_path())

        results = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            # 只处理以 "py utils.py " 开头的行
            if line.startswith("py utils.py "):
                cmd_part = line[len("py utils.py "):]
                try:
                    args_list = shlex.split(cmd_part)
                except Exception as e:
                    results.append(f"第 {line_num} 行命令解析失败: {e}")
                    continue

                if not args_list:
                    results.append(f"第 {line_num} 行命令为空")
                    continue

                tool_name = args_list[0]
                tool_args = args_list[1:]

                # 执行工具
                try:
                    module = importlib.import_module(f"utils.{tool_name}")
                except ImportError:
                    results.append(f"第 {line_num} 行: 未知工具 '{tool_name}'")
                    continue

                if not hasattr(module, "run"):
                    results.append(f"第 {line_num} 行: 工具 '{tool_name}' 缺少 run 函数")
                    continue

                try:
                    result = module.run(ctx, tool_args)
                    if result is not None:
                        results.append(f"第 {line_num} 行: {result}")
                    else:
                        results.append(f"第 {line_num} 行: 执行成功（无输出）")
                except Exception as e:
                    results.append(f"第 {line_num} 行: 执行出错 - {e}")
            else:
                # 忽略不以 py utils.py 开头的行
                # 可选择记录调试信息，此处忽略
                pass

        # 打印所有结果
        for res in results:
            print(res)
        sys.exit(0)

    # 命令行模式（有参数）
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()
    args = sys.argv[2:]

    if command == "help":
        if args:
            show_help(args[0])
        else:
            show_help()
        sys.exit(0)

    # 正常工具调用
    tool_name = command
    tool_args = args

    try:
        module = importlib.import_module(f"utils.{tool_name}")
    except ImportError:
        print(f"错误：未知的工具名称 '{tool_name}'")
        print("可用工具：")
        utils_dir = os.path.join(os.path.dirname(__file__), "utils")
        if os.path.isdir(utils_dir):
            for f in os.listdir(utils_dir):
                if f.endswith(".py") and f not in ("__init__.py", "core.py"):
                    print(f"    {f[:-3]}")
        sys.exit(1)

    if not hasattr(module, "run"):
        print(f"错误：工具模块 '{tool_name}' 缺少 run 函数")
        sys.exit(1)

    from utils import core
    ctx = core.Context(core.load_root_path())

    try:
        result = module.run(ctx, tool_args)
        if result is not None:
            print(result)
    except Exception as e:
        print(f"执行工具时出错：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()