根据您提供的 `CommandRule` 代码以及现有的文件列表，主要问题在于 **入口文件 `utils.py` 如何与 `utils/` 目录下的具体工具文件进行交互**。

`CommandRule` 的逻辑是调用 `python utils.py <工具名> <参数...>`。因此，我们需要一个位于根目录（或 `utils_py_path` 指向的位置）的 **分发器 (`utils.py`)**，以及位于 `utils/` 文件夹下的 **核心定义 (`core.py`)** 和 **具体工具**。

以下是重构后的代码结构和内容，完全适配您的 `CommandRule`。

### 1. 目录结构规范

确保您的目录结构如下所示：

```text
ProjectRoot/
├── agent.py            # 主程序
├── utils.py            # [关键] 工具分发入口
├── .env                # 配置文件 (ROOT_PATH)
└── utils/              # 工具包目录
    ├── __init__.py     # 空文件，标识为包
    ├── core.py         # [关键] 上下文与路径验证
    ├── read.py
    ├── write.py
    ├── list.py
    └── ... (其他工具)
```

---

### 2. 核心分发器：`utils.py`

这是最重要的文件。它负责接收 `CommandRule` 传来的命令行参数，加载 `utils/` 下对应的模块，并注入上下文。

```python
# [START] UTILS-ENTRY
# version: 002
# 上下文：被 CommandRule 作为子进程拉起。先决调用：CommandRule.match_and_execute。后续调用：具体工具模块的 run 方法。
# 输入参数：sys.argv (命令行参数)
# 输出参数：标准输出 (工具执行结果)
import sys
import os
import importlib

# 确保能引用到 utils 包
sys.path.append(os.getcwd())

def main():
    # 1. 参数校验
    if len(sys.argv) < 2:
        print("Usage: py utils.py <tool_name> [args...]")
        return

    tool_name = sys.argv[1]
    tool_args = sys.argv[2:]

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
```

---

### 3. 工具核心：`utils/core.py`

用于安全地处理路径，防止 AI 访问项目之外的文件。

```python
# [START] UTILS-CORE
# version: 001
# 上下文：被 utils.py 导入，用于创建 Context 对象。
import os
import re

class Context:
    def __init__(self, root_path):
        self.root_path = os.path.abspath(root_path)

    def validate_path(self, relative_path):
        """
        验证并返回绝对路径，防止路径穿越 (../)
        """
        if not relative_path:
            raise ValueError("路径不能为空")
            
        # 简单防穿越
        if '..' in relative_path:
            raise ValueError(f"非法路径（包含 '..'）：{relative_path}")
            
        # 拼接并获取绝对路径
        full_path = os.path.abspath(os.path.join(self.root_path, relative_path))
        
        # 再次确认最终路径是否在 root_path 下
        if not full_path.startswith(self.root_path):
            raise ValueError(f"访问拒绝：路径 {relative_path} 超出工作目录范围")
            
        return full_path

def load_root_path():
    """从环境变量或 .env 获取根路径"""
    root = os.getcwd()
    env_path = os.path.join(root, '.env')
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('ROOT_PATH='):
                        val = line.split('=', 1)[1].strip()
                        return val.strip('"').strip("'")
        except:
            pass
    return root
# [END] UTILS-CORE
```

---

### 4. 具体工具实现 (适配新接口)

以下是根据您列出的文件，适配后的几个关键工具。请替换 `utils/` 目录下对应的文件。

#### `utils/read.py`
```python
# [START] TOOL-READ
# version: 002
# 描述：读取文件内容
import os

def run(ctx, args):
    if not args:
        return "Error: read 需要至少一个文件路径"
    
    results = []
    for rel_path in args:
        try:
            full_path = ctx.validate_path(rel_path)
            if not os.path.exists(full_path):
                results.append(f"=== {rel_path} ===\nError: 文件不存在")
                continue
                
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            # 添加文件名标记，方便 AI 区分多个文件
            results.append(f"=== {rel_path} ===\n{content}\n=== end of {rel_path} ===")
        except Exception as e:
            results.append(f"=== {rel_path} ===\nError: {str(e)}")
            
    return "\n\n".join(results)
# [END] TOOL-READ
```

#### `utils/write.py`
```python
# [START] TOOL-WRITE
# version: 002
# 描述：覆盖写入文件
import os

def run(ctx, args):
    if len(args) < 2:
        return "Error: write 需要 2 个参数：<路径> <内容...>"
    
    rel_path = args[0]
    # 将剩余参数重新拼接为内容 (处理空格)
    content = " ".join(args[1:])
    
    try:
        full_path = ctx.validate_path(rel_path)
        # 自动创建父目录
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"Success: 已写入 {rel_path} (长度: {len(content)})"
    except Exception as e:
        return f"Error: 写入失败 - {str(e)}"
# [END] TOOL-WRITE
```

#### `utils/list.py`
```python
# [START] TOOL-LIST
# version: 002
# 描述：列出目录文件
import os

def run(ctx, args):
    rel_path = args[0] if args else "."
    
    try:
        full_path = ctx.validate_path(rel_path)
        if not os.path.isdir(full_path):
            return f"Error: {rel_path} 不是一个目录"
            
        items = os.listdir(full_path)
        result = []
        for item in items:
            item_path = os.path.join(full_path, item)
            suffix = "/" if os.path.isdir(item_path) else ""
            result.append(f"{item}{suffix}")
            
        return "\n".join(result) if result else "(空目录)"
    except Exception as e:
        return f"Error: 无法列出目录 - {str(e)}"
# [END] TOOL-LIST
```

#### `utils/git.py`
```python
# [START] TOOL-GIT
# version: 002
# 描述：执行 git 命令
import subprocess

def run(ctx, args):
    if not args:
        return "Error: git 需要子命令 (e.g., status, diff)"
    
    # 限制只能运行 git 命令
    cmd = ["git"] + args
    
    try:
        # 在 ctx.root_path 下执行
        result = subprocess.run(
            cmd,
            cwd=ctx.root_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr
            
        return output.strip()
    except subprocess.TimeoutExpired:   
        return "Error: Git 命令超时"
    except Exception as e:
        return f"Error: Git 执行失败 - {str(e)}"
# [END] TOOL-GIT
```

### 总结修改点

1.  **废弃 `command_executor.py`**：因为新的 `CommandRule` 已经直接使用了 `asyncio.create_subprocess_exec`，不再需要旧的执行器。
2.  **新增根目录 `utils.py`**：这是关键。`CommandRule` 调用的是 `py utils.py`，所以必须在根目录有一个入口脚本来解析参数并分发给 `utils/` 下的子模块。
3.  **统一接口**：所有 `utils/*.py` 文件现在都统一使用 `run(ctx, args)` 签名，并且返回字符串供 `CommandRule` 捕获输出。
4.  **上下文隔离**：通过 `ctx.validate_path` 确保文件操作限制在工作目录内。