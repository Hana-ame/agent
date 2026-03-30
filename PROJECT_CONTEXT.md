# 项目上下文记录 - Hana-ame/agent (client分支)

## 项目结构
```
adapter-api/
├── adapters/
│   ├── base.py          # MasterClient基类
│   ├── deepseek.py      # DeepSeek适配器
│   └── deepseek_webapp.py # DeepSeek Web UI适配器（主要）
├── agent.py             # 主客户端入口
├── plugins/
│   ├── base.py         # 插件基类
│   └── prompt.py       # Prompt相关插件
└── .gitignore
```

## 核心文件分析

### 1. `adapters/deepseek_webapp.py`
- 继承自`MasterClient`
- 解析DeepSeek Web UI的WebSocket响应
- 状态管理: THINK, RESPONSE, FINISHED
- 事件机制和消息队列

### 2. `agent.py`
- 主入口: `async def main()`
- 连接到WebSocket服务器 (`ws://127.26.3.1:8080/ws/client`)
- 插件架构: 支持SaveCodePlugin, RunBashCodeBlock, DefaultPrompt, LogPlugin
- 循环对话模式

### 3. `adapters/base.py`
- `MasterClient`基类
- WebSocket消息发送/接收包装
- 浏览器配对管理 (`available_browsers`, `paired`)

## WebSocket协议
```
发送格式:
{
  "channel": "system" | "client",
  "payload": {...}
}

系统命令:
- "system": {"command": "list"}           # 获取可用浏览器
- "system": {"command": "pair", ...}      # 配对浏览器
- "client": {"command": "match", ...}     # 导航到域名
- "client": {"command": "send_prompt", ...} # 发送消息
- "client": {"command": "new_chat"}       # 新建对话
```

## 当前任务需求
1. **修改客户端支持OpenAI API格式**
   - 接收OpenAI风格请求: `{"model": "deepseek-chat", "messages": [...]}`
   - 处理`[NEW_CHAT]`系统消息
   - 返回OpenAI兼容响应

2. **添加相似度判断**
   - 新消息与历史对话的相似度计算
   - 低于阈值时自动发送`new_chat`命令
   - 保持上下文连贯性

3. **推送到原仓库**
   - 分支: `client` 基础上的修改
   - 新分支名: `openai-api-adapter`
   - 推送到 `git@github.com:Hana-ame/agent.git`

## 相似度算法设计
```python
def calculate_similarity(text1, text2):
    """简单Jaccard相似度"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union

# 阈值设定
NEW_CHAT_THRESHOLD = 0.2  # 相似度低于20%视为全新话题
```

## OpenAI API兼容格式
```json
// 请求
{
  "model": "deepseek-chat",
  "messages": [
    {"role": "system", "content": "[NEW_CHAT]"},
    {"role": "user", "content": "Hello"}
  ],
  "stream": false
}

// 响应
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "deepseek-chat",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }]
}
```

## 修改计划
1. **新建文件** `api/openai_server.py`
   - HTTP服务器接收OpenAI格式请求
   - 调用现有DeepSeekWebApp客户端

2. **增强现有客户端**
   - 添加相似度计算和历史管理
   - 自动`new_chat`决策

3. **保持向后兼容**
   - 原有CLI接口不变
   - 新增HTTP API接口

## 待确认问题
- WebSocket服务器地址: `ws://127.26.3.1:8080/ws/client`
- 是否需要多客户端支持？
- 相似度阈值如何确定？
- 是否支持流式响应？