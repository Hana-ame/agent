# [START] UTILS-ENTRY
import sys
import os
import importlib

sys.path.append(os.getcwd())

def main():
    if len(sys.argv) < 2:
        print("Usage: py utils.py <tool_name> [args...]")
        return

    # 修复：直接信任 sys.argv。
    # 因为 CommandExecutor 已经使用了 shlex.split 并通过 subprocess 传递列表，
    # 操作系统会自动处理好参数边界，sys.argv[1:] 已经是解析好的工具名和参数。
    parsed_args = sys.argv[1:]

    tool_name = parsed_args[0]
    tool_args = parsed_args[1:]

    try:
        from utils import core
        ctx = core.Context(core.load_root_path())
    except Exception as e:
        print(f"Error: 初始化上下文失败: {e}")
        return

    try:
        module_name = f"utils.{tool_name}"
        module = importlib.import_module(module_name)
        if hasattr(module, "run") and callable(module.run):
            result = module.run(ctx, tool_args)
            if result is not None:
                print(result)
        else:
            print(f"Error: 工具 '{tool_name}' 缺少 run 方法")
    except ImportError:
        print(f"Error: 找不到工具 '{tool_name}'")
    except Exception as e:
        print(f"Error: 执行异常: {e}")

if __name__ == "__main__":
    main()
# [END] UTILS-ENTRY