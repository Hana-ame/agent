# [START] UTILS-PKG
# version: 001
# 上下文：工具调用主入口脚本。先决调用：无。后续调用：动态加载环境初始化。
# 输入参数：无
# 输出参数：无
import sys
import importlib.util
import os
import shlex
import json

ALIASES = {
    'cat': 'read',
    'ls': 'list',
    'mv': 'move',
    'rm': 'delete',
}
# [END] UTILS-PKG

# [START] UTILS-LOADER
# version: 001
# 上下文：尝试在指定路径集合中寻址并加载外部独立的 Python 工具文件。先决调用：接收到外部工具名调用请求。后续调用：通过执行工具内的 run 方法运行。
# 输入参数：tool_name (str)
# 输出参数：编译并挂载的 Python module 对象或 None
def load_tool_module(tool_name: str):
    root_path = os.getcwd()
    search_paths =[
        os.path.join(root_path, ".agent"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
    ]
    
    for path in search_paths:
        mod_path = os.path.join(path, f"{tool_name}.py")
        if os.path.isfile(mod_path):
            spec = importlib.util.spec_from_file_location(f"tool_{tool_name}", mod_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None
#[END] UTILS-LOADER

# [START] UTILS-SCANNER
# version: 001
# 上下文：获取当前系统两级目录下的所有可用工具集合。先决调用：用户通过 help 指令查看列表。后续调用：交由 help 格式化打印。
# 输入参数：无
# 输出参数：去重后的所有工具纯名称列表 (List[str])
def get_all_tool_names():
    root_path = os.getcwd()
    search_paths =[
        os.path.join(root_path, ".agent"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
    ]
    tools = set()
    for path in search_paths:
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith(".py") and f not in ("__init__.py", "core.py"):
                    tools.add(f[:-3])
    return sorted(list(tools))
# [END] UTILS-SCANNER

# [START] UTILS-INFO
# version: 001
# 上下文：探测单个工具详情及说明。先决调用：UTILS-LOADER 寻址工具模块。后续调用：帮助系统的 UI 打印。
# 输入参数：tool_name (str)
# 输出参数：描述字段组成的字典 (dict)
def get_tool_info(tool_name: str):
    actual_tool = ALIASES.get(tool_name, tool_name)
    info = {
        'name': actual_tool,
        'alias': tool_name if actual_tool != tool_name else None,
        'description': '无说明',
        'doc': None
    }
    mod = load_tool_module(actual_tool)
    if mod:
        if mod.__doc__:
            doc = mod.__doc__.strip()
            info['doc'] = doc
            lines = doc.split('\n')
            desc = lines[0].strip()
            if not desc and len(lines) > 1:
                desc = lines[1].strip()
            info['description'] = desc
    else:
        info['description'] = f"工具 '{actual_tool}' 不存在"
    return info
# [END] UTILS-INFO

# [START] UTILS-SHOW-HELP
# version: 001
# 上下文：解析外部 help 入参并输出文档结果。先决调用：启动脚本被传入 help 指令。后续调用：流出到标准输出供查看。
# 输入参数：args (list)
# 输出参数：无
def show_help(args: list):
    json_output = '--json' in args
    
    if len(args) > 0 and not args[0].startswith('--'):
        tool_name = args[0]
        info = get_tool_info(tool_name)
        if json_output:
            print(json.dumps(info, ensure_ascii=False, indent=2))
        else:
            print(info['doc'] if info['doc'] else f"工具 '{tool_name}' 没有文档。")
        return

    tools = get_all_tool_names()
    tools_info =[get_tool_info(t) for t in tools]

    if json_output:
        print(json.dumps({info['name']: info for info in tools_info}, ensure_ascii=False, indent=2))
    else:
        if not tools_info:
            print("没有找到匹配的工具。")
            return
        for info in tools_info:
            alias_str = f" (别名: {info['alias']})" if info['alias'] else ""
            print(f"  {info['name']:15}{alias_str} - {info['description']}")
        print("\n常用别名：cat(→read), ls(→list), mv(→move), rm(→delete)")
# [END] UTILS-SHOW-HELP

#[START] UTILS-MAIN
# version: 001
# 上下文：工具代理脚本独立生命周期起点。先决调用：被 CommandRule 作为独立进程拉起。后续调用：加载具体工具并传递假冒上下文 Context。
# 输入参数：无
# 输出参数：无
def main():
    if len(sys.argv) < 2:
        print("错误: 必须指定工具名称。")
        sys.exit(1)

    command = sys.argv[1].lower()
    args = sys.argv[2:]

    if command == "help":
        show_help(args)
        sys.exit(0)

    raw_tool = command
    tool_name = ALIASES.get(raw_tool, raw_tool)

    mod = load_tool_module(tool_name)
    if not mod:
        print(f"错误：未知的工具名称 '{raw_tool}'。尝试 'py utils.py help' 了解可用工具。")
        sys.exit(1)

    if not hasattr(mod, "run"):
        print(f"错误：工具模块 '{tool_name}' 缺少 run 函数")
        sys.exit(1)

    class DummyContext:
        def validate_path(self, path):
            return path
            
    try:
        result = mod.run(DummyContext(), args)
        if result is not None:
            print(result)
    except Exception as e:
        print(f"执行工具时出错：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
# [END] UTILS-MAIN