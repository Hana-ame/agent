# [START] UTILS-ENTRY
import sys
import os
import importlib

# 修复：动态提取当前脚本所在的目录，无论从哪里运行，都能正确定位 utils
utils_dir = os.path.dirname(os.path.abspath(__file__))
# 找出包的父级目录 (以便能够成功执行 from utils import core)
parent_dir = os.path.dirname(utils_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# 兜底：保留 cwd 在环境变量中，兼容可能有其他依赖项在当前目录的情况
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

def main():
    if len(sys.argv) < 2:
        print("Usage: py utils.py <tool_name> [args...]")
        return

    # 因为 Executor 使用了 shlex(posix=True) 传参，sys.argv 已经干干净净没有外层引号了
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