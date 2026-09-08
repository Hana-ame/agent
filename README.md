# Vertex-Edge Agent Framework 使用说明

Vertex-Edge Agent Framework 是一个数据驱动型 AI Agent 编排框架，通过**顶点（Vertex）**承载状态与数据，通过**边（Edge）**承载业务逻辑与大模型推理。

---

## 一、安装与环境准备

- **Python 版本要求**：`Python 3.12+`

```bash
# 1. 安装项目依赖
pip install -r requirements.txt

# 2. 本地安装包
pip install -e .
```

---

## 二、快速上手

### 1. 方式 A：通过 JSON 配置编排工作流（推荐）

#### 步骤 1：编写业务脚本 (`workflow/trans.py`)

```python
def to_uppercase(content: str, settings: dict, staging: dict) -> str:
    """将输入内容转换为大写。"""
    return content.upper()

def add_signature(content: str, settings: dict, staging: dict) -> str:
    """为内容追加署名。"""
    author = settings.get("author", "VEA")
    return f"{content}\n-- Processed by {author}"
```

#### 步骤 2：定义工作流清单 (`workflow/graph.json`)

```json
{
  "version": "4.0",
  "session_id": "demo_pipeline",
  "metadata": {"name": "Text Processing Pipeline"},
  "vertices": [
    {"name": "input_node", "content": "hello world", "attributes": ["start"], "state": "data ready"},
    {"name": "upper_node", "content": "", "state": "todo"},
    {"name": "output_node", "content": "", "attributes": ["end"], "state": "todo"}
  ],
  "edges": [
    {
      "id": "e_upper",
      "input_vertex": "input_node",
      "output_vertex": "upper_node",
      "script": "workflow/trans.py:to_uppercase"
    },
    {
      "id": "e_sign",
      "input_vertex": "upper_node",
      "output_vertex": "output_node",
      "script": "workflow/trans.py:add_signature",
      "settings": {"author": "VertexEdgeAgent"}
    }
  ]
}
```

#### 步骤 3：加载并运行

```python
import asyncio
from framework import VertexStoreV4, DiscreteGraphLoaderV4, ExecutorV4

async def main():
    store = VertexStoreV4(":memory:")
    graph = DiscreteGraphLoaderV4.load_from_manifest("workflow/graph.json")
    DiscreteGraphLoaderV4.populate_store(graph, store)

    executor = ExecutorV4(graph=graph, store=store, max_concurrency=2)
    result = await executor.run()

    print("执行状态:", result.success)
    final_node = store.get_vertex(graph.session_id, "output_node")
    print("输出内容:\n", final_node.content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 2. 方式 B：通过 Python 代码编排工作流

```python
import asyncio
from framework import (
    VertexStoreV4, VertexRecordV4, VertexStateV4, VertexAttributeV4,
    GraphV4, ExecutorV4, CodeEdgeV4
)

async def main():
    store = VertexStoreV4(":memory:")
    session_id = "code_sess"
    graph = GraphV4(session_id=session_id)

    # 1. 添加顶点
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="src",
        content="Antigravity",
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value
    ))
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="dst",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value
    ))

    # 2. 绑定转换边
    def reverse_text(data, settings, staging):
        return data[::-1]

    graph.add_edge(CodeEdgeV4("edge_rev", "src", "dst", script=reverse_text))
    graph.validate()

    # 3. 运行执行器
    executor = ExecutorV4(graph=graph, store=store)
    result = await executor.run()

    print("结果:", store.get_vertex(session_id, "dst").content)  # 输出: ytivargitnA

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 三、大模型推理管道 (SenseNova)

框架开箱即用支持商汤 SenseNova 6.8 远程大模型推理（可选择通过环境变量 `SENSENOVA_API_KEY` 设置密钥）：

```python
import asyncio
import json
from framework import (
    VertexStoreV4, VertexRecordV4, VertexStateV4, VertexAttributeV4,
    GraphV4, ExecutorV4, SenseNovaEdgeV4, CodeEdgeV4
)

async def main():
    store = VertexStoreV4(":memory:")
    session_id = "llm_sess"
    graph = GraphV4(session_id=session_id)

    # 1. 提示词节点
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="prompt_node",
        content="计算 15 + 25。请严格输出 JSON: {\"result\": 40}，不要包含任何多余文字。",
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value
    ))

    # 2. 模型推理输出节点（声明 JSON 校验属性）
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="model_node",
        content="",
        attributes=[VertexAttributeV4.JSON.value],
        state=VertexStateV4.TODO.value
    ))

    # 3. 最终结果节点
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="final_node",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value
    ))

    # 4. 连接 SenseNova 边与后处理计算边
    graph.add_edge(SenseNovaEdgeV4(
        edge_id="e_llm",
        input_vertex="prompt_node",
        output_vertex="model_node",
        settings={"temperature": 0.1}
    ))

    def parse_result(content, settings, staging):
        data = json.loads(content)
        return f"计算得到结果: {data['result']}"

    graph.add_edge(CodeEdgeV4("e_calc", "model_node", "final_node", script=parse_result))
    graph.validate()

    executor = ExecutorV4(graph=graph, store=store)
    res = await executor.run()

    print("终点内容:", store.get_vertex(session_id, "final_node").content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 四、核心特性用法

### 1. 多源汇聚合并策略 (Fan-In Merge)

当多个上游边汇入同一个节点时，使用 `MergeStrategyV4` 指定合并策略：

```python
from framework import MergeStrategyV4

# 覆盖（默认）
store.apply_merge_strategy(sess, "target", incoming, strategy=MergeStrategyV4.OVERWRITE)

# 合并字典: {**existing, **incoming}
store.apply_merge_strategy(sess, "target", '{"b": 2}', strategy=MergeStrategyV4.JSON_MERGE)

# 列表追加: [item1, item2, ...]
store.apply_merge_strategy(sess, "target", '"new_item"', strategy=MergeStrategyV4.LIST_APPEND)

# 自定义归约函数
def custom_reducer(existing: str, incoming: str) -> str:
    return f"{existing},{incoming}".strip(",")

store.apply_merge_strategy(sess, "target", "item", strategy=MergeStrategyV4.REDUCER_SCRIPT, reducer_fn=custom_reducer)
```

---

### 2. 异常重试自愈与熔断保护 (ReflexiveEdge)

节点执行发生异常时置为 `REJECT`，自环边响应并重试：

```python
from framework import ReflexiveEdgeV4, VertexStateV4

def recovery_fn(old_content, settings, staging):
    return f"{old_content} (请简化回答)"

reflexive_edge = ReflexiveEdgeV4(
    edge_id="e_retry",
    vertex_name="worker_node",
    trigger_state=VertexStateV4.REJECT.value,
    target_state=VertexStateV4.TODO_URGENT.value,
    max_retries=3,
    script=recovery_fn
)
```

若重试超过 `max_retries`，节点置为 `FORBIDDEN` 熔断锁死。

---

### 3. 嵌套子图

在节点的 `attributes` 中添加 `subgraph`，并在 `content` 中配置子图清单路径：

```json
{
  "name": "subgraph_box",
  "content": "{\"subgraph_manifest\": \"sub_workflow/graph.json\"}",
  "attributes": ["subgraph"],
  "state": "todo"
}
```

执行器会自动挂接子图桥接边并递归执行。

---

## 五、服务化运行 (Web 仪表盘与 API)

### 1. 启动服务

```bash
uvicorn framework.server_v4:app --host 0.0.0.0 --port 8000
```

### 2. Web 仪表盘

打开浏览器访问：`http://localhost:8000/dashboard`

- 实时查看节点 DAG 拓扑网络与状态高亮（绿: DATA_READY / 黄: TODO / 蓝: TODO_URGENT / 红: REJECT / 灰: FORBIDDEN）。
- 点击节点查看最新内容、状态与执行计数。

### 3. API 调用示例

- **运行指定会话工作流**：
  ```bash
  curl -X POST http://localhost:8000/api/sessions/{session_id}/run \
       -H "Content-Type: application/json" \
       -d '{"max_concurrency": 4}'
  ```

- **SSE 流式执行**：
  ```bash
  curl -N http://localhost:8000/api/sse/execute \
       -H "Content-Type: application/json" \
       -d '{"session_id": "sess_1", "manifest_path": "workflow/graph.json"}'
  ```

---

## 六、运行测试

```bash
# 运行离线测试套件
python -m pytest tests/ -v -m "not live"

# 运行全量测试套件
python -m pytest tests/ -v
```

---

## 七、文档归档

历史架构设计规范、版本审查记录与排查分析报告已统一归档至 [docs/archive/](docs/archive/) 目录。
