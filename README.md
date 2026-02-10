# DeepSeek CLI Chat 工具

一个支持**流式思考链**、**多会话管理**和**文档引用**的 LLM 命令行对话工具。

---

## 核心特性

| 特性 | 说明 |
|:---|:---|
| 🔥 **实时流式渲染** | 区分显示"思考过程"（黄色）与"正式回答"（绿色） |
| 💬 **多会话隔离** | 每个会话独立 JSON 文件，随时切换上下文 |
| 📄 **文档引用** | 自动包裹文档内容，支持代码/论文/日志分析 |
| 🧠 **Context/Payload 双模式** | 灵活加载系统提示（外部文件 or 本地配置） |
| 🎨 **智能颜色输出** | 自动检测 TTY，支持 `--no-color` 禁用 |
| ⚡ **原子化持久化** | 对话历史安全写入，防损坏 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install requests
```

### 2. 配置 API 端点

创建 `endpoint.json`：
```json
{
    "endpoint": "https://api.deepseek.com/v1/chat/completions",
    "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
}
```

### 3. 配置对话模板（可选）

创建 `payload.json`：
```json
{
    "model": "deepseek-reasoner",
    "temperature": 0.7,
    "messages": [],
    "_context": ""
}
```

> `_context` 字段会被自动指向当前会话历史文件，无需手动修改。

### 4. 启动对话

```bash
python chat.py
```

---

## 命令行用法

### 基础交互模式

```bash
# 默认使用 history.json
python chat.py

# 指定会话文件（项目隔离）
python chat.py -c project_a.json
python chat.py -c project_b.json
```

### 单次查询模式

```bash
# 直接提问
python chat.py "解释量子纠缠"

# 带文档分析
python chat.py -d paper.pdf "总结这篇论文的方法论"

# 全新会话（不加载历史）
python chat.py "开始新话题" --new
```

### 交互模式内置命令

| 命令 | 作用 |
|:---|:---|
| `exit` / `quit` | 退出程序 |
| `/reset` / `/clear` | 清空当前会话历史 |
| `/new` | 强制开启新对话轮次 |

---

## 完整参数说明

```bash
python chat.py --help

# 输出：
usage: chat.py [-h] [-d PATH] [-c FILE] [--endpoint FILE] [--payload FILE]
               [--new] [--no-color] [message]

DeepSeek CLI 对话工具 - 支持流式思考链与多会话管理

positional arguments:
  message               直接发送的消息内容（不提供则进入交互模式）

optional arguments:
  -h, --help            显示帮助信息并退出
  -d PATH, --document PATH
                        附加参考文档路径（将包裹在标记中发送）
  -c FILE, --context FILE
                        指定历史记录 JSON 文件（默认: history.json）
  --endpoint FILE       覆盖 API 配置文件（默认: endpoint.json）
  --payload FILE        覆盖 Payload 模板文件（默认: payload.json）
  --new                 开启全新对话，清空指定的 context 文件
  --no-color            禁用 ANSI 颜色输出
```

---

## 典型使用场景

### 场景 1：代码审查会话

```bash
# 创建专用会话
python chat.py -c code_review.json -d src/main.py "分析这段代码的潜在问题"

# 继续同一会话深入讨论
python chat.py -c code_review.json "给出优化建议"

# 查看历史
cat code_review.json | jq '.[].content'
```

### 场景 2：论文阅读助手

```bash
# 每篇论文独立会话
python chat.py -c paper_gpt4.json -d gpt4.pdf "总结核心贡献" --new
python chat.py -c paper_rag.json -d rag_survey.pdf "对比几种检索方法" --new
```

### 场景 3：调试日志分析

```bash
# 直接粘贴大段日志
python chat.py -d /var/log/error.log "分析错误模式"

# 或交互式逐步排查
python chat.py
> /clear
> 我看到一个报错：Connection refused, errno 111
```

---

## 文件结构说明

```
workspace/
├── endpoint.json          # API 配置（endpoint + api_key）
├── payload.json           # 默认参数模板（model, temperature 等）
├── history.json           # 默认会话历史（自动创建）
├── project_a.json         # 自定义会话 A
├── project_b.json         # 自定义会话 B
└── chat.py                # 主程序
```

### 会话历史格式

```json
[
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助？", "reasoning_content": "用户打招呼..."},
    {"role": "user", "content": "解释深度学习"},
    {"role": "assistant", "content": "深度学习是...", "reasoning_content": "用户询问技术概念..."}
]
```

---

## 进阶配置

### 自定义 Payload 模板（Context 模式）

修改 `payload.json` 指向外部系统提示：

```json
{
    "model": "deepseek-reasoner",
    "_context": "./prompts/coder_prompt.json",
    "temperature": 0.5
}
```

创建 `prompts/coder_prompt.json`：

```json
[
    {"role": "system", "content": "你是资深代码审查专家，擅长发现安全漏洞..."},
    {"role": "user", "content": "示例：如何防止 SQL 注入？"},
    {"role": "assistant", "content": "使用参数化查询..."}
]
```

此时发送请求会自动加载外部提示作为基础上下文。

---

## 架构设计

```
┌─────────────────────────────────────────┐
│           ChatCLI (chat.py)             │
│  - 终端交互 / 命令解析 / 状态管理        │
├─────────────────────────────────────────┤
│      JsonPayloadSender (逻辑层)          │
│  - 消息组装 / 文件管理 / 双模式策略      │
├─────────────────────────────────────────┤
│      JsonApiRequester (网络层)           │
│  - HTTP 请求 / SSE 流式 / Hook 机制      │
└─────────────────────────────────────────┘
```

---

## 故障排查

| 问题 | 解决方案 |
|:---|:---|
| `配置文件缺失` | 检查 `endpoint.json` 和 `payload.json` 是否存在 |
| `API 返回 401` | 确认 `api_key` 有效且未过期 |
| `无颜色输出` | 检查终端是否支持 ANSI，或强制 `--no-color` 再试 |
| `历史文件损坏` | 自动重置为空，原文件需手动备份恢复 |
| `文档过大` | 自动截断至 100KB，建议分段发送 |

---

## 许可证

MIT