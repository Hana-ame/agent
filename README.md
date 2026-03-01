
# Auto-Agent

一个基于多轮对话的自动化任务执行系统，支持通过 LLM 代理（Agent）调用本地工具来完成文件操作、代码修改等任务。

## 项目特点

- **多轮对话循环**：LLM 与用户通过工具调用持续交互，直到任务完成
- **WebSocket 连接**：基于 WebSocket 的 DeepSeek 桥接协议，支持实时流式响应
- **安全的文件操作**：所有文件修改基于已有文件，避免直接生成全新内容，减少幻觉风险
- **工具化架构**：模块化的工具系统，易于扩展
- **人工干预机制**：支持暂停/恢复机制，用户可在关键点介入

## 项目结构

```
.
├── main.py              # 主入口，启动 Agent 循环
├── run.py               # 便捷的启动脚本
├── agent.py             # 简化版启动器（可能已弃用）
├── llm_client.py        # LLM 客户端，WebSocket 桥接
├── expand.py            # 文本扩展/处理工具
├── utils.py             # 工具调用入口和分发器
├── utils/               # 工具模块目录
│   ├── core.py          # 工具上下文和根路径管理
│   ├── read.py          # 读取文件内容
│   ├── write.py         # 写入文件（基于上一轮输出）
│   ├── append.py        # 追加内容到文件
│   ├── replace.py       # 替换文件中的文本
│   ├── list.py          # 列出目录内容
│   ├── delete.py        # 删除文件
│   ├── pause.py         # 暂停任务，等待人工干预
│   ├── resume.py        # 恢复暂停的任务
│   ├── memory.py        # 持久化存储（SQLite）
│   ├── git.py           # Git 操作
│   └── write_multiple.py # 批量写入多个文件
├── agent/               # Agent 配置目录
│   └── profiles.json    # LLM 连接配置文件（含 endpoint）
├── SYSTEM_PROMPT.txt    # 系统提示词，定义 Agent 行为规则
├── MESSAGE.txt          # 用户输入消息（轮询读取）
├── LAST_RESPONSE.txt    # 保存上一轮 LLM 的输出
├── .env                 # 环境配置（ROOT_PATH 等）
└── memory.db            # SQLite 持久化数据库
```

## 快速开始

### 1. 环境配置

创建 `.env` 文件指定工作目录：

```bash
# .env
ROOT_PATH="C:/your/project/path"
```

### 2. 配置 LLM 连接

编辑 `agent/profiles.json`：

```json
{
  "deepseek": {
    "endpoint": "wss://your-bridge-url/ws/client"
  }
}
```

### 3. 启动 Agent

```bash
# 通过命令行参数
python main.py "你的任务描述"

# 或通过消息文件
echo "你的任务描述" > MESSAGE.txt
python run.py
```

## 核心工作流程

1. **初始化**：读取 `SYSTEM_PROMPT.txt` 和用户输入
2. **发送**：将消息发送给 LLM（WebSocket）
3. **接收**：获取 LLM 的推理过程（reasoning）和回复内容（content）
4. **解析**：提取回复中的工具命令（`py utils.py ...`）
5. **执行**：执行第一个找到的命令
6. **循环**：将工具输出作为下一轮输入，重复 2-5 步
7. **结束**：检测到 `=== FINISH ===` 标记时停止

## 安全写入模式

为避免 LLM 幻觉生成错误代码，项目采用**两轮写入**机制：

1. **第一轮**：LLM 输出完整的代码内容（自动保存到 `LAST_RESPONSE.txt`）
2. **第二轮**：LLM 输出 `py utils.py write [文件路径]` 指令，将 `LAST_RESPONSE.txt` 的内容写入目标文件

这确保了：
- 代码内容先经过验证/查看
- 写入操作明确且可控
- 避免回复中的说明文字污染代码文件

## 人工干预机制

- **自动暂停**：当工具输出以 `PAUSED:` 开头时，系统进入暂停状态
- **创建暂停标志**：`.pause` 文件作为暂停标记
- **用户介入**：在暂停期间，用户可查看状态、修改文件、在 `MESSAGE.txt` 中输入指令
- **恢复执行**：删除 `.pause` 文件后，Agent 自动继续

## 日志记录

所有交互记录保存在 `agent.log`，包括：
- 每轮的发送内容
- LLM 的推理过程
- LLM 的回复内容
- 工具执行输出

## 命令规则

Agent 在回复中输出命令时需遵循：

1. 命令以 `py utils.py` 开头
2. 多行命令中**只有第一行**会被执行
3. 需要写文件时，先输出纯代码内容，再输出 write 指令
4. 任务完成时输出 `=== FINISH ===` 标记

## 工具列表

| 工具 | 用途 |
|------|------|
| `read` | 读取文件内容 |
| `write` | 将 LAST_RESPONSE.txt 写入指定路径 |
| `append` | 追加内容到文件末尾 |
| `replace` | 替换文件中的指定文本 |
| `list` | 列出目录内容 |
| `delete` | 删除文件 |
| `pause` | 暂停等待人工干预 |
| `resume` | 恢复暂停的任务 |
| `memory` | 持久化存储/查询 |
| `git` | 执行 Git 命令 |
| `write_multiple` | 批量写入多个文件 |

**注意**：工具的返回格式已经过优化，避免包含“成功”等可能干扰LLM推理的词语。具体地：
- `write`：执行成功后返回被写入文件的内容原文，后跟一行“写入至：[文件路径]”。
- `write_multiple`：执行成功后返回每个文件的块，格式为：
  ```
  === [文件路径] ===
  [文件内容原文]
  写入至：[文件路径]
  ===
  ```
  多个文件块之间用两个换行分隔。