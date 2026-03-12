# [START] UTILS-ENTRY
# version: 0.0.3
# 上下文：被 CommandRule 作为子进程拉起。先决调用：CommandRule.match_and_execute。后续调用：具体工具模块的 run 方法。
# 输入参数：sys.argv (命令行参数)
# 输出参数：标准输出 (工具执行结果)
import sys
import os
import importlib
import shlex

# 确保能引用到 utils 包
sys.path.append(os.getcwd())

def main():
    # 1. 参数校验
    if len(sys.argv) < 2:
        print("Usage: py utils.py <tool_name>[args...]")
        return

    # 修复上层（如 CommandRule）粗暴按空格切分导致带引号参数被破坏的问题
    # 正常 Shell 传入 sys.argv 时引号已被剥离；如果发现残留引号，说明被错误切片了
    if any('"' in arg or "'" in arg for arg in sys.argv[1:]):
        raw_args_str = " ".join(sys.argv[1:])
        try:
            parsed_args = shlex.split(raw_args_str)
        except ValueError:
            # 遇到未闭合的引号等解析失败情况，回退使用原始 sys.argv
            parsed_args = sys.argv[1:]
    else:
        parsed_args = sys.argv[1:]

    if not parsed_args:
        print("Usage: py utils.py <tool_name> [args...]")
        return

    tool_name = parsed_args[0]
    tool_args = parsed_args[1:]

    # 2. 初始化上下文 (依赖 utils/core.py)
    try:
        from utils import core
        # 加载根路径，默认为当前工作目录
        root_path = core.load_root_path()
        ctx = core.Context(root_path)
    except ImportError as e:
        print(f"Error: 无法加载核心模块 utils.core: {e}")
        return
    except Exception as e:
        print(f"Error: 初始化上下文失败: {e}")
        return

    # 3. 动态加载工具模块
    try:
        # 尝试从 utils 包中加载对应的模块
        module_name = f"utils.{tool_name}"
        module = importlib.import_module(module_name)
    except ImportError:
        print(f"Error: 找不到工具 '{tool_name}' (utils/{tool_name}.py 不存在)")
        return

    # 4. 执行工具的 run 方法
    if hasattr(module, "run") and callable(module.run):
        try:
            result = module.run(ctx, tool_args)
            # 统一输出结果到 stdout，供 CommandRule 捕获
            if result is not None:
                print(result)
        except Exception as e:
            print(f"Error: 执行工具 '{tool_name}' 时发生异常: {e}")
    else:
        print(f"Error: 工具 '{tool_name}' 未定义 'run(ctx, args)' 入口函数")

if __name__ == "__main__":
    main()
# [END] UTILS-ENTRY