现在我已经收集了足够的信息来编写README。让我基于对项目的理解来撰写 README.md 的内容：

```markdown
# Auto-Agent

一个基于多轮对话的自动化任务执行系统，支持通过 LLM 代理（Agent）调用本地工具来完成文件操作、代码修改等任务。

## 项目特点

- **多轮对话循环**：LLM 与用户通过工具调用持续交互，直到任务完成
- **双桥接支持**：同时支持 WebSocket（DeepSeek 私有协议）和 HTTP（OpenAI 兼容 API）
- **安全的文件操作**：所有文件修改基于已有文件，避免直接生成全新内容，减少幻觉风险
- **工具化架构**：模块化的工具系统，易于扩展
- **人工干预机制**：支持暂停/恢复机制，用户可在关键点介入

## 项目结构

```
.
├── main.py              # 主入口，启动 Agent 循环
├── run.py               # 便捷的启动脚本
├── agent.py             # 简化版启动器（可能已弃用）
├── llm_client.py        # LLM 客户端，支持 WebSocket 和 HTTP 桥接
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
├── payloads/            # API 请求配置模板
│   └── kimi.json        # Kimi API 的 payload 配置
├── agent/               # Agent 配置目录
│   └── profiles.json    # LLM 连接配置文件（含 endpoint 和 api_key）
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
  "kimi": {
    "endpoint": "https://api.moonshot.cn/v1/chat/completions",
    "api_key": "your-api-key-here",
    "model": "Pro/moonshotai/Kimi-K2.5"
  },
  "deepseek": {
    "endpoint": "wss://your-deepseek-bridge/ws/client"
  }
}
```

### 3. 配置 Payload（HTTP 模式）

在 `payloads/` 目录下创建 JSON 文件，例如 `kimi.json`：

```json
{
  "model": "Pro/moonshotai/Kimi-K2.5",
  "temperature": 1,
  "max_tokens": 1024,
  "top_p": 0.95
}
```

### 4. 启动 Agent

```bash
# WebSocket 模式（DeepSeek）
python main.py "wss://your-bridge-url/ws/client"

# HTTP 模式（OpenAI 兼容 API）
python main.py kimi
```

或通过消息文件启动：

```bash
echo "你的任务描述" > MESSAGE.txt
python run.py
```

## 核心工作流程

1. **初始化**：读取 `SYSTEM_PROMPT.txt` 和用户输入
2. **发送**：将消息发送给 LLM
3. **接收**：获取 LLM 的推理过程和回复内容
4. **解析**：提取回复中的工具命令（`py utils.py ...`）
5. **执行**：执行第一个找到的命令
6. **循环**：将工具输出作为下一轮输入，重复 2-5 步
7. **结束**：检测到 `=== FINISH ===` 标记时停止

## 特殊机制

### 安全写入模式

为避免 LLM 幻觉生成错误代码，项目采用**两轮写入**机制：

1. **第一轮**：LLM 输出完整的代码内容（保存到 `LAST_RESPONSE.txt`）
2. **第二轮