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

为了方便，部分常用命令设有别名：
    cat  -> read
    ls   -> list
    mv   -> move
    rm   -> delete

使用 'py utils.py help' 查看所有工具。
使用 'py utils.py help <工具名>' 查看具体工具的详细帮助。
使用 'py utils.py help --json' 以 JSON 格式输出工具列表。
使用 'py utils.py help --search <关键词>' 搜索工具。
"""

import sys
import importlib
import os
import shlex
import json

# 别名映射：用户输入的命令 -> 实际工具名称
ALIASES = {
    'cat': 'read',
    'ls': 'list',
    'mv': 'move',
    'rm': 'delete',
}

def get_tool_info(tool_name):
    """获取指定工具的详细信息，返回字典包含 name, description, doc, alias"""
    actual_tool = ALIASES.get(tool_name, tool_name)
    info = {
        'name': actual_tool,
        'alias': tool_name if actual_tool != tool_name else None,
        'description': '无说明',
        'doc': None
    }
    try:
        module = importlib.import_module(f"utils.{actual_tool}")
        if module.__doc__:
            doc = module.__doc__.strip()
            info['doc'] = doc
            # 提取第一行或第一段作为简短描述
            lines = doc.split('\n')
            desc = lines[0].strip()
            if not desc and len(lines) > 1:
                desc = lines[1].strip()
            info['description'] = desc
    except ImportError:
        info['description'] = f"工具 '{actual_tool}' 不存在或无法加载"
    return info

def list_tools(json_output=False, search_term=None):
    """列出所有可用工具，支持 JSON 输出和搜索"""
    utils_dir = os.path.join(os.path.dirname(__file__), "utils")
    tools = []
    if os.path.isdir(utils_dir):
        for f in os.listdir(utils_dir):
            if f.endswith(".py") and f not in ("__init__.py", "core.py"):
                tools.append(f[:-3])
    tools.sort()

    # 收集所有工具信息
    tools_info = []
    for t in tools:
        info = get_tool_info(t)  # 传入实际工具名，获取原始信息
        # 如果提供了搜索词，检查是否匹配
        if search_term:
            search_lower = search_term.lower()
            if (search_lower in info['name'].lower() or
                search_lower in info['description'].lower() or
                (info['doc'] and search_lower in info['doc'].lower())):
                tools_info.append(info)
        else:
            tools_info.append(info)

    if json_output:
        # 输出 JSON 格式
        output = {info['name']: info for info in tools_info}
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        # 普通文本格式
        if not tools_info:
            print("没有找到匹配的工具。")
            return
        for info in tools_info:
            alias_str = f" (别名: {info['alias']})" if info['alias'] else ""
            print(f"  {info['name']:15}{alias_str} - {info['description']}")
        # 显示别名映射总览
        if not search_term:
            print("\n常用别名：cat(→read), ls(→list), mv(→move), rm(→delete)")

def show_tool_help(tool_name, json_output=False):
    """显示单个工具的详细帮助"""
    info = get_tool_info(tool_name)
    if json_output:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        if info['doc']:
            print(info['doc'])
        else:
            print(f"工具 '{tool_name}' 没有提供帮助文档。")

def show_help(args=None):
    """显示帮助信息，args 是命令行剩余参数列表"""
    if args is None:
        args = []

    # 解析参数（简单手动解析）
    json_output = False
    search_term = None
    tool_name = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--json':
            json_output = True
        elif arg == '--search' and i+1 < len(args):
            search_term = args[i+1]
            i += 1
        elif not arg.startswith('--'):
            # 第一个非选项参数视为工具名
            tool_name = arg
        i += 1

    if tool_name:
        # 显示特定工具帮助
        show_tool_help(tool_name, json_output)
    elif search_term is not None:
        # 搜索工具
        list_tools(json_output, search_term)
    else:
        # 显示所有工具列表
        list_tools(json_output, search_term)

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

                raw_tool = args_list[0]
                tool_name = ALIASES.get(raw_tool, raw_tool)  # 别名映射
                tool_args = args_list[1:]

                # 执行工具
                try:
                    module = importlib.import_module(f"utils.{tool_name}")
                except ImportError:
                    results.append(f"第 {line_num} 行: 未知工具 '{raw_tool}'（映射后为 '{tool_name}' 仍不存在）")
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
        show_help(args)
        sys.exit(0)

    # 正常工具调用，应用别名映射
    raw_tool = command
    tool_name = ALIASES.get(raw_tool, raw_tool)
    tool_args = args

    try:
        module = importlib.import_module(f"utils.{tool_name}")
    except ImportError:
        print(f"错误：未知的工具名称 '{raw_tool}'（映射后为 '{tool_name}' 仍不存在）")
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