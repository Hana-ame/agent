
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils.py - 工具调用入口

用法：
    py utils.py <工具名称> [参数...]

工具列表（位于 utils/ 目录下）：
    read <相对路径>
    write <相对路径> <内容>
    append <相对路径> <内容>
    replace <相对路径> <旧文本> <新文本>
    list <相对路径>
    delete <相对路径>
    ... 可扩展
"""

import sys
import importlib
import os

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    tool_name = sys.argv[1].lower()
    tool_args = sys.argv[2:]

    # 尝试导入工具模块
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

    # 检查模块是否包含 run 函数
    if not hasattr(module, "run"):
        print(f"错误：工具模块 '{tool_name}' 缺少 run 函数")
        sys.exit(1)

    # 从 core 导入上下文
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