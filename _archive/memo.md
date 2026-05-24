# Memo

## 项目概况
- FastAPI 项目，监听端口 8000
- 公网访问地址：`http://wsl-8000.moonchan.xyz`

## 免费模型列表 (仅限自动处理使用)
- siliconflow-cn/Qwen/Qwen3-8B
- siliconflow-cn/Qwen/Qwen3.5-4B
- opencode/big-pickle
- opencode/deepseek-v4-flash-free
- opencode/minimax-m2.5-free
- opencode/nemotron-3-super-free
- opencode/qwen3.6-plus-free
- google/gemma-4-31b-it
- google/gemini-3-flash-preview

**规则：在自动处理（Node Worker 或 Auto666 任务）中，禁止使用上述列表之外的模型。**

## 本地测试
- 使用 `curl -x ""` 绕过代理，例如：`curl -x "" http://localhost:8000/api/...`

## 规则
- `html/` 文件夹中的每次修改，都要上传到 `upload.moonchan.xyz`
- `html/readme.md` 记录所有上传链接
- 所有新建的 `.html` 文件都必须包含可折叠的 API Endpoint 输入框
- **`html/` 内文件每次修改后，必须重新 upload 并更新 `readme.md`**

## Node 设计
- `accept_tags` 为 AND 逻辑：必须**所有** tag 都有未处理数据，节点才执行
- `{tag_name}` 在 prompt 中作为变量，会被替换为对应 tag 的文本值
- `output_tag` 为单个输出 tag

## 文件结构
- `app.py` — FastAPI 后端（纯 CRUD + API，不含轮询逻辑）
- `runner.py` — 独立的 Node 轮询引擎，仅通过 `state.db` 与 FastAPI 通信
- `pipeline_exec.py` — Pipeline Chain 的同步执行器
- `automation_agent.py` — Board 666 自动化 Agent
- `html/index.html` — 导航入口
- `html/prompt.html` — Prompt 管理页
- `html/node.html` — Node 配置页
- `html/exec.html` — 执行日志页
- `html/ping.html` — 连接测试页
- `html/pipeline.html` — Pipeline Chain 管理页（LangGraph 模拟）
- `html/readme.md` — 上传记录
- `create_db.py` — 独立数据库初始化脚本（`python create_db.py --seed`）

## Pipeline Chain (LangGraph 模拟)

Pipeline Chain 是同步执行的 Prompt 链，模拟 LangGraph 的线性执行模式。

### JSON Schema
每个 Pipeline 定义为一个 JSON，包含 `entry` 和 `steps` 列表：

```json
{
  "entry": "analyzer",
  "steps": [
    {
      "id": "analyzer",
      "name": "分析器",
      "model": "opencode/deepseek-v4-flash-free",
      "prompt": "分析：{user_input}",
      "output_key": "analysis",
      "next": "responder"
    },
    {
      "id": "responder",
      "name": "响应器",
      "model": "opencode/minimax-m2.5-free",
      "prompt": "回复：{analysis}",
      "output_key": "response",
      "next": null
    }
  ]
}
```

### 执行逻辑
1. 从 `entry` 指定的步骤开始
2. 每个步骤：替换 `{变量}` → 调用 `opencode run` → 存入 `output_key` → 移到 `next`
3. 遇到 `next: null` 或错误时停止

### 数据库表
- `pipeline`：id, name, definition(JSON), created_at, updated_at
- `pipeline_exec`：id, pipeline_id, input_data, output_data, status, error, steps_log, created_at

### API 端点
- `GET/POST /api/pipelines` — 列表/创建
- `GET/PUT/DELETE /api/pipelines/{id}` — CRUD
- `POST /api/pipelines/{id}/run` — 执行（body: `{"input_data": {...}}`）
- `GET /api/pipelines/{id}/execs` — 执行历史
- `GET /api/pipeline-execs` — 全部执行历史

### 与 Node Worker 的区别
- **Node Worker**：异步轮询，节点独立扫描 tag，适用于事件驱动、多路并行
- **Pipeline Chain**：同步执行，步骤顺序执行、变量传递，适用于确定性流程

