# Subgraph & 离散配置加载使用说明

本示例展示了在 Vertex-Edge Agent Framework v4 中如何通过独立 JSON 配置文件与文件夹加载顶点（Vertex）和边（Edge），以及如何构建和运行嵌套子图（Subgraph）。

---

## 1. 快速运行

进入代码仓库根目录，执行以下命令：

```bash
python3 examples/subgraph_v4/run.py
```

---

## 2. 目录结构

```
examples/subgraph_v4/
├── parent_in.json                # 父图起始 Vertex 配置
├── parent_subgraph.json          # 子图容器 Vertex 配置（指向 child_graph.json）
├── parent_out.json               # 父图结束 Vertex 配置
├── e_start_to_subgraph.json      # 父图连接边（输入 -> 子图）
├── e_subgraph_to_output.json     # 父图连接边（子图 -> 输出）
├── parent_graph.json             # 父图 Master 清单文件
├── run.py                        # 可直接运行的演示脚本
├── child/                        # 子图配置与脚本目录
│   ├── child_start.json          # 子图起始 Vertex
│   ├── child_middle.json         # 子图中间 Vertex
│   ├── child_end.json            # 子图结束 Vertex
│   ├── child_edge1.json          # 子图内部边 1（执行 cleaner.py）
│   ├── child_edge2.json          # 子图内部边 2（执行 enricher.py）
│   ├── child_graph.json          # 子图清单文件
│   ├── cleaner.py                # 文本清洗处理脚本
│   └── enricher.py               # 数据丰富处理脚本
└── discrete_dir_demo/            # 文件夹批量加载演示目录
    ├── vertices/                 # 存放顶点 JSON
    │   ├── v1_in.json
    │   └── v2_out.json
    └── edges/                    # 存放边 JSON
        └── e1_dir.json
```

---

## 3. 使用方式

### 方式一：直接指定单个 JSON 文件路径加载

`graph.add_vertex()` 和 `graph.add_edge()` 支持直接传入 `.json` 配置文件路径：

```python
from framework.graph_v4 import GraphV4

graph = GraphV4(session_id="my_session")

# 指定 JSON 文件位置加载 Vertex
graph.add_vertex("examples/subgraph_v4/parent_in.json")
graph.add_vertex("examples/subgraph_v4/parent_subgraph.json")
graph.add_vertex("examples/subgraph_v4/parent_out.json")

# 指定 JSON 文件位置加载 Edge
graph.add_edge("examples/subgraph_v4/e_start_to_subgraph.json")
graph.add_edge("examples/subgraph_v4/e_subgraph_to_output.json")
```

#### Vertex JSON 配置格式示例 (`parent_in.json`)：
```json
{
  "name": "session_input",
  "content": "hello subgraph world",
  "attributes": ["start"],
  "state": "data_ready"
}
```

#### Edge JSON 配置格式示例 (`e_start_to_subgraph.json`)：
```json
{
  "id": "e_in_to_subgraph",
  "type": "code",
  "input_vertex": "session_input",
  "output_vertex": "enrichment_box"
}
```

---

### 方式二：指定文件夹一键批量加载

支持指定包含顶点和边配置的文件夹进行加载：

```python
from framework.graph_v4 import GraphV4

# 1. 直接通过类方法从文件夹加载新图
graph = GraphV4.from_directory("examples/subgraph_v4/discrete_dir_demo", session_id="my_session")

# 2. 或在已有图实例中加载文件夹
existing_graph = GraphV4(session_id="existing_session")
existing_graph.load_directory("examples/subgraph_v4/discrete_dir_demo")

# 3. 也可以分别把文件夹路径传给 add_vertex / add_edge
existing_graph.add_vertex("examples/subgraph_v4/discrete_dir_demo/vertices")
existing_graph.add_edge("examples/subgraph_v4/discrete_dir_demo/edges")
```

文件夹支持以下结构：
- 包含 `vertices/`（或 `nodes/`）与 `edges/` 子目录。
- 扁平目录（自动根据 JSON 中的字段识别 Vertex 与 Edge）。
- 包含 `graph.json` 或 `manifest.json` 清单文件的目录。

---

### 方式三：嵌套子图（Subgraph）配置与执行

在父图中定义一个属性包含 `"subgraph"` 的节点，其 `content` 指定子图清单路径或子图目录：

#### 子图节点配置 (`parent_subgraph.json`)：
```json
{
  "name": "enrichment_box",
  "attributes": ["subgraph"],
  "state": "todo",
  "content": {
    "subgraph_manifest": "child/child_graph.json"
  }
}
```

子图内部定义起始节点（`attributes: ["start"]`）和结束节点（`attributes: ["end"]`）。

通过 `SSEExecutorV4` 执行流程时，框架会自动解析嵌套子图并挂载执行桥接：

```python
from framework.graph_v4 import DiscreteGraphLoaderV4
from framework.server_v4 import SessionGraphManagerV4
from framework.sse_executor_v4 import SSEExecutorV4
from framework.vertex_v4 import VertexStoreV4

store = VertexStoreV4(":memory:")
manager = SessionGraphManagerV4(store)

executor = SSEExecutorV4(
    manager=manager,
    store=store,
    default_manifest="examples/subgraph_v4/parent_graph.json",
)

result = await executor.execute_harness_call()
```
