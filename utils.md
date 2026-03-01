# Utils - 安全、可扩展的命令行工具集

这是一个基于 Python 的模块化命令行工具集，设计用于通过简单的命令行调用执行常见文件操作，同时具备**路径安全防护**（防止 `../` 逃逸）和**易于扩展**的架构。非常适合作为 AI 代理（如 LLM）的操作接口，或日常脚本工具箱。

## ✨ 特性
- **模块化设计**：每个工具独立为一个 Python 文件，位于 `utils/` 文件夹下，新增工具只需添加文件。
- **安全沙箱**：所有路径操作都基于 `.env` 中设置的 `ROOT_PATH` 根目录，自动阻止访问根目录之外的路径。
- **即插即用**：通过 `py utils.py <工具名> [参数...]` 直接调用，无需额外配置。
- **UTF-8 支持**：统一使用 UTF-8 编码读写文件，避免乱码。
- **可扩展性**：轻松集成 Git、压缩、网络请求等更多功能。

## 📦 安装
### 1. 环境要求
- Python 3.6+
- 仅需标准库，无需第三方依赖。

### 2. 获取脚本
将以下文件/文件夹结构放置在你的工作目录中：
```
your_project/
├── utils.py              # 主入口
├── .env                  # (可选) 配置文件
└── utils/                # 工具模块文件夹
    ├── __init__.py       # 空文件
    ├── core.py           # 核心模块（路径验证、配置加载）
    ├── read.py           # 读取文件
    ├── write.py          # 写入文件
    ├── append.py         # 追加内容
    ├── replace.py        # 替换文本
    ├── list.py           # 列出目录
    └── delete.py         # 删除文件
```
你可以从项目仓库下载，或根据文档手动创建。

## ⚙️ 配置（可选）
在脚本所在目录创建 `.env` 文件，设置安全根目录 `ROOT_PATH`：
```ini
ROOT_PATH=/home/user/projects
```
- 所有文件操作将限制在此目录内，任何试图访问外部路径（如 `../etc/passwd`）的操作都会被拒绝。
- 若未设置 `.env` 或未定义 `ROOT_PATH`，默认使用当前工作目录作为根目录。

## 🚀 使用方法
### 基本格式
```bash
py utils.py <工具名> [参数...]
```
（Linux/macOS 下可能需用 `python3 utils.py` 代替 `py utils.py`）

### 内置工具列表
#### 1. 读取文件 `read`
```bash
py utils.py read 相对路径
```
示例：
```bash
py utils.py read docs/notes.txt
```
输出文件内容（UTF-8）。

#### 2. 写入文件 `write`（覆盖原内容）
```bash
py utils.py write 相对路径 "要写入的内容"
```
示例：
```bash
py utils.py write config/settings.json '{"debug": true}'
```

#### 3. 追加内容 `append`
```bash
py utils.py append 相对路径 "追加的内容"
```
示例：
```bash
py utils.py append logs/access.log "192.168.1.1 - - [01/Mar/2025:12:00:00]"
```

#### 4. 替换文件中的文本 `replace`
```bash
py utils.py replace 相对路径 "旧文本" "新文本"
```
示例：
```bash
py utils.py replace src/main.py "DEBUG=True" "DEBUG=False"
```

#### 5. 列出目录内容 `list`
```bash
py utils.py list 相对路径
```
示例：
```bash
py utils.py list src
```
输出每行一个文件名/目录名。

#### 6. 删除文件 `delete`
```bash
py utils.py delete 相对路径
```
示例：
```bash
py utils.py delete temp/old.log
```

> **注意**：所有路径参数均为**相对于根目录**的路径。例如根目录为 `/home/user/proj`，则 `read docs/file.txt` 实际读取 `/home/user/proj/docs/file.txt`。

## 🧩 扩展指南：添加新工具
想要添加新功能（如 Git 操作、压缩文件等）非常容易，只需两步：

1. 在 `utils/` 文件夹下新建一个 `.py` 文件，例如 `git.py`。
2. 在该文件中实现 `run(ctx, args)` 函数：
   - `ctx`：上下文对象，包含 `ctx.root_path`（根目录绝对路径）和 `ctx.validate_path(user_path)` 方法（用于安全验证路径）。
   - `args`：命令行参数列表（不包括工具名本身）。
   - 函数应返回一个字符串，该字符串将打印到标准输出（作为命令结果）。

**示例：实现一个简单的 Git 状态工具 `utils/git.py`**
```python
import subprocess

def run(ctx, args):
    # 安全：如果 git 命令需要操作文件，可用 ctx.validate_path 检查
    # 例如：path = ctx.validate_path(args[0])  # 假设第一个参数是相对路径
    try:
        # 在根目录下执行 git 命令
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=ctx.root_path
        )
        # 合并 stdout 和 stderr 作为返回
        return result.stdout + result.stderr
    except Exception as e:
        return f"Git 错误：{e}"
```
完成后即可调用：
```bash
py utils.py git status
py utils.py git log --oneline -5
```

**安全提示**：若新工具涉及文件操作，务必使用 `ctx.validate_path(用户提供的路径)` 获取安全的绝对路径，该函数会在路径越界时抛出 `ValueError`。

## 🔒 安全机制
- **根目录锁定**：所有路径先与根目录拼接，再通过 `os.path.abspath` 规范化，最后检查是否以根目录绝对路径开头。若以 `..` 等方式试图逃逸，操作被拒绝并提示错误。
- **异常处理**：文件不存在、权限不足等异常均被捕获并返回友好错误信息，不会导致脚本崩溃。

## 📝 常见问题
### Q: 路径中包含空格怎么办？
A: 用双引号将路径或内容括起来，例如：
```bash
py utils.py write "my notes/file.txt" "Hello world"
```

### Q: 如何查看当前根目录？
A: 可在 `utils.py` 中取消注释 `print(f"当前根目录: {root}", file=sys.stderr)` 一行，或添加一个临时工具如 `utils/root.py` 输出 `ctx.root_path`。

### Q: 可以同时在多个项目中使用不同的根目录吗？
A: 可以。每个项目只需在其目录下放置自己的 `.env` 文件，设置对应的 `ROOT_PATH` 即可。

## 🤝 贡献
欢迎添加更多实用工具！请遵循模块化设计，确保每个工具文件独立且包含 `run` 函数，并尽量使用 `ctx.validate_path` 保证安全。

## 📄 许可证
MIT