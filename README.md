# DeepSeek Web UI Automation Client

一个通过WebSocket控制DeepSeek Web UI的自动化客户端，现已支持OpenAI兼容API。

## 📦 项目结构

```
├── adapters/                    # 适配器层
│   ├── base.py                 # MasterClient基类
│   ├── deepseek.py             # DeepSeek适配器
│   └── deepseek_webapp.py      # DeepSeek Web UI适配器（主要）
├── agent.py                    # 主客户端入口（CLI）
├── openai_server.py            # OpenAI兼容HTTP服务器（NEW）
├── openai_adapter.py           # OpenAI适配器核心（NEW）
├── plugins/                    # 插件系统
│   ├── base.py                # 插件基类
│   └── prompt.py              # Prompt相关插件
└── tests/                      # 测试文件
```

## 🚀 快速开始

### 1. 原始CLI用法（保持不变）
```bash
# 连接到WebSocket服务器
python agent.py --message="你好，请帮我写代码" --new-chat

# 或指定自定义WebSocket URL
python agent.py ws://your-server:8080/ws/client --message="Hello"
```

### 2. OpenAI API模式（新增功能）
```bash
# 启动HTTP服务器（零依赖）
python openai_server.py --port=8000

# 发送OpenAI兼容请求
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Hello"}]}'
```

## 🔌 WebSocket协议

连接到 `wss://moonchan.publicvm.com/ws/client`（默认），支持以下操作：

### 核心命令
```python
# 获取可用浏览器列表
await client.send("system", {"command": "list"})

# 与浏览器配对
await client.send("system", {
    "command": "pair",
    "title": browser_title,
    "type": "browser",
    "timestamp": timestamp
})

# 导航到DeepSeek聊天
await client.call_match("chat.deepseek.com")

# 发送聊天消息
await client.call_send_prompt(text="你的消息")

# 新建对话（清空历史）
await client.call_new_chat()
```

## 🧠 OpenAI兼容API

### 支持的端点
```
GET  /health                # 健康检查
GET  /v1/models             # 列出可用模型
POST /v1/chat/completions   # 聊天补全（支持流式）
```

### 请求示例
```json
{
  "model": "deepseek-chat",
  "messages": [
    {"role": "system", "content": "[NEW_CHAT]"},
    {"role": "user", "content": "你好，写一个Python排序函数"}
  ],
  "stream": false
}
```

### 智能特性
1. **智能相似度路由** - 自动判断是否发送 `new_chat`
2. **多会话管理** - 支持 `client_id` 会话隔离
3. **闲置清理** - 5分钟无活动自动关闭连接
4. **错误处理** - 完整的错误响应和重试机制

## ⚡ 相似度算法

```python
# Jaccard相似度公式
相似度 = 交集词数 / 并集词数

# 决策规则
if 相似度 < 0.20:
    发送 new_chat    # 全新话题
else:
    复用现有会话     # 相关话题
```

### 示例
```
用户消息1: "如何写Python爬虫？"
用户消息2: "Python requests库怎么用？"
相似度: ~45% → 复用会话（相关话题）

用户消息1: "解释一下量子计算"  
用户消息2: "推荐一家好吃的餐厅"
相似度: ~0% → 发送 new_chat（完全不相关）
```

## 🔧 与OpenClaw集成

在OpenClaw配置中添加：
```json
"models": {
  "providers": {
    "deepseek-webui": {
      "baseUrl": "http://localhost:8000/v1",
      "apiKey": "dummy-key",
      "auth": "api-key",
      "api": "openai-completions",
      "authHeader": true,
      "models": [{
        "id": "deepseek-chat",
        "name": "DeepSeek Web UI (Browser Automation)",
        "api": "openai-completions",
        "reasoning": false,
        "input": ["text"],
        "contextWindow": 32768
      }]
    }
  }
}
```

## 🛠️ 插件系统

### 内置插件
- `DefaultPrompt` - 默认提示词生成
- `SaveCodePlugin` - 保存代码块到文件
- `RunBashCodeBlock` - 执行bash代码块
- `LogPlugin` - 日志记录

### 自定义插件
继承 `Plugin` 基类，实现以下方法：
```python
class CustomPlugin(Plugin):
    async def before_prompt(self, args, req):
        # 在发送prompt前执行
        return False  # 返回True表示需要重新处理
    
    async def after_prompt(self, args, req, resp):
        # 在收到响应后执行
        return False  # 返回True表示需要重新处理
```

## 📊 性能特性

### 会话管理
- **连接池** - 复用WebSocket连接
- **智能清理** - 5分钟闲置自动关闭
- **多路复用** - 支持并行多客户端
- **错误恢复** - 自动重连和配对

### 资源优化
- **内存限制** - 对话历史最多20条
- **快速响应** - 异步处理和相似度计算
- **扩展性** - 支持负载均衡和多浏览器

## 🐛 故障排除

### 常见问题
1. **WebSocket连接失败**
   ```bash
   # 检查服务器是否运行
   netstat -tlnp | grep :8080
   
   # 或使用不同URL
   python agent.py ws://your-server:8080/ws/client
   ```

2. **OpenAI API无法访问**
   ```bash
   # 检查HTTP服务器
   curl http://localhost:8000/health
   
   # 查看日志
   python openai_server.py --port=8001
   ```

3. **相似度阈值调整**
   ```python
   # 在 openai_adapter.py 中修改
   NEW_CHAT_THRESHOLD = 0.15  # 改为15%
   ```

### 日志级别
```bash
# 查看详细日志
DEBUG=1 python openai_server.py --port=8000
```

## 🔮 未来计划

### 近期计划
- [ ] 流式响应支持
- [ ] Prometheus指标暴露
- [ ] Docker容器化部署
- [ ] 配置文件支持

### 功能增强
- [ ] 语义相似度（词向量）
- [ ] 动态阈值调整
- [ ] 浏览器负载均衡
- [ ] 会话持久化

## 📄 许可证

MIT License - 自由使用和修改

## 🤝 贡献

欢迎提交Issue和Pull Request！

1. Fork仓库
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📞 支持

- GitHub Issues: 报告问题
- Pull Requests: 代码贡献
- Discussions: 功能讨论

---

**核心优势**：将DeepSeek Web UI自动化包装为标准OpenAI API，实现智能相似度路由和会话管理，同时保持原有CLI的向后兼容性。