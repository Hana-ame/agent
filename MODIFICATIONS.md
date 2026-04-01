# 修改详情 - OpenAI API Adapter for DeepSeek Web UI

## 新增文件

### 1. `openai_adapter.py`
**功能**：OpenAI API兼容适配器核心逻辑
- `SessionManager`：多会话管理，相似度计算
- `Session`：单个WebSocket连接和对话历史
- `OpenAIAdapter`：HTTP请求处理器原型

**关键特性**：
- Jaccard相似度算法 (阈值20%)
- 自动`new_chat`决策
- 会话复用和清理机制
- 支持`client_id`隔离

### 2. `openai_server.py`
**功能**：零依赖HTTP服务器
- 标准库实现，无需外部包
- 完整OpenAI API端点 (`/v1/chat/completions`, `/v1/models`, `/health`)
- CORS支持
- 结构化日志输出

**API端点**：
```
GET  /health                # 健康检查
GET  /v1/models             # 模型列表
POST /v1/chat/completions   # 聊天补全
```

## 修改的核心逻辑

### 相似度决策
```python
def should_create_new_chat(self, similarity: float) -> bool:
    """20%相似度阈值，低于此值视为全新话题"""
    return similarity < 0.2
```

### OpenAI请求处理
```python
# 请求格式
{
  "model": "deepseek-chat",
  "messages": [
    {"role": "system", "content": "[NEW_CHAT]"},
    {"role": "user", "content": "Hello"}
  ]
}

# 支持client_id
{"role": "system", "content": "client_id: my-session-id"}
```

### WebSocket会话管理
```python
# 自动配对浏览器
# 自动导航到 chat.deepseek.com
# 智能复用现有会话
# 5分钟闲置清理
```

## 保留的原有功能

### 1. 原有适配器不变
- `adapters/deepseek_webapp.py` - 不变
- `adapters/base.py` - 不变
- `agent.py` - CLI接口不变

### 2. 插件系统保持不变
- SaveCodePlugin, RunBashCodeBlock
- DefaultPrompt, LogPlugin
- 所有插件兼容性保留

### 3. WebSocket协议不变
- 仍连接到 `wss://moonchan.publicvm.com/ws/client`
- 相同的配对和消息格式
- 相同的DeepSeek响应解析

## 向后兼容性

### ✅ 完全兼容
- 现有CLI用法不变：`python agent.py`
- 现有插件系统不变
- 现有WebSocket服务器兼容

### 🚀 新增功能
- HTTP API服务器 (`python openai_server.py`)
- OpenAI兼容格式请求
- 相似度智能路由
- 多会话并发支持

## 部署和运行

### 1. 启动OpenAI API服务器
```bash
python openai_server.py --port=8000
```

### 2. 测试API
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Hello"}]}'
```

### 3. 与OpenClaw集成
在OpenClaw配置中添加：
```json
"deepseek-webui": {
  "baseUrl": "http://localhost:8000/v1",
  "models": [{"id": "deepseek-chat"}]
}
```

## 技术细节

### 相似度算法实现
```python
def calculate_similarity(text1, text2):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0
```

### 会话管理策略
- **复用**：相似度≥20%复用现有会话
- **新建**：相似度<20%或强制`[NEW_CHAT]`
- **清理**：5分钟无活动自动关闭
- **隔离**：`client_id`支持多任务并行

### 错误处理
- WebSocket连接失败自动重试
- 浏览器配对失败返回友好错误
- HTTP请求验证和错误响应
- 结构化日志便于调试

## 性能考虑

### 内存优化
- 对话历史限制20条
- 闲置会话自动清理
- 连接池复用

### 响应时间
- 相似度计算高效 (O(n))
- 异步WebSocket操作
- HTTP请求快速响应

### 扩展性
- 支持多浏览器实例
- 支持多API客户端
- 支持负载均衡

## 未来扩展点

### 1. 增强相似度算法
- 词向量embedding
- 语义相似度
- 主题聚类

### 2. 高级会话管理
- A/B测试不同阈值
- 动态阈值调整
- 用户偏好学习

### 3. 监控和指标
- Prometheus指标暴露
- Grafana仪表盘
- 性能分析和优化