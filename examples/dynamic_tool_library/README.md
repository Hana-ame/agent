# 🧩 Dynamic Tool Library & Subgraph Dispatcher (动态工具库与子图路由)

本示例展示了如何将多个**子图作为独立工具库**，结合**分配节点（Router / Dispatcher）**在运行时根据任务意图动态从磁盘加载、热拼接并执行匹配的工具子图（方案 A）。

---

## 🏗️ 架构设计

```text
                             ┌───────────────────────┐
                             │     v_user_query      │ (用户任务输入)
                             └───────────┬───────────┘
                                         │
                                  [classify_intent]
                                         │
                                         ▼
                             ┌───────────────────────┐
                             │       Router          │ (意图分类判断)
                             └───────────┬───────────┘
                                         │ 动态从 tools/ 目录选择并加载
                     ┌───────────────────┼───────────────────┐
                     ▼                   ▼                   ▼
            [code_analyzer.json] [finance_calculator.json] [data_extractor.json]
                     │                   │                   │
                     └───────────────────┼───────────────────┘
                                         │ graph.insert_subgraph() 动态热拼接
                                         ▼
                     ┌───────────────────────────────────────┐
                     │   Spliced Tool Subgraph Execution     │
                     │   sub_in ➔ e_step1 ➔ e_step2 ➔ sub_out│
                     └───────────────────┬───────────────────┘
                                         │ (bridge_out)
                                         ▼
                             ┌───────────────────────┐
                             │    v_final_output     │ (最终聚合输出)
                             └───────────────────────┘
```

---

## 📂 目录结构

```text
examples/dynamic_tool_library/
├── tools/                        # 🧰 工具子图库目录（每个工具都是一个合法的完整子图）
│   ├── code_analyzer.json        # 工具 1：代码静态扫描与审查子图
│   ├── finance_calculator.json   # 工具 2：财务指标与利润率计算子图
│   └── data_extractor.json       # 工具 3：非结构化文本联系方式与实体抽取子图
├── tool_scripts.py               # 业务逻辑与意图分类函数
├── demo.py                       # 端到端运行主脚本
└── README.md                     # 本说明文档
```

---

## 🚀 核心实现步骤

### 1. 工具库独立配置（如 `tools/code_analyzer.json`）
每个工具子图具有自己的起始节点（`sub_in`）和终点节点（`sub_out`），独立维护：
```json
{
  "vertices": [
    { "name": "sub_in", "state": "todo", "attributes": ["start"] },
    { "name": "sub_linted", "state": "todo" },
    { "name": "sub_out", "state": "todo", "attributes": ["end"] }
  ],
  "edges": [
    { "id": "e_lint", "type": "code", "input_vertex": "sub_in", "output_vertex": "sub_linted", "script": "..." },
    { "id": "e_critique", "type": "code", "input_vertex": "sub_linted", "output_vertex": "sub_out", "script": "..." }
  ]
}
```

### 2. 运行时动态识别与热拼接 (`demo.py`)
```python
# 1. 意图判断
tool_name = classify_intent(task_query)
tool_path = f"tools/{tool_name}.json"

# 2. 从工具库加载子图
tool_subgraph = DiscreteGraphLoaderV4.load_from_manifest(tool_path)

# 3. 热拼接到主图，桥接输入和输出
graph.insert_subgraph(
    subgraph=tool_subgraph,
    incoming_bindings={"v_user_query": "sub_in"},
    outgoing_bindings={"sub_out": "v_final_output"},
    name_prefix=tool_name,
)

# 4. 同步至 SQLite 并由执行器运行
DiscreteGraphLoaderV4.populate_store(graph, store)
executor = ExecutorV4(graph=graph, store=store, snapshot_dir="snapshots")
result = await executor.run()
```

---

## 🏃 运行验证

在仓库根目录下运行：
```bash
python3 examples/dynamic_tool_library/demo.py
```
