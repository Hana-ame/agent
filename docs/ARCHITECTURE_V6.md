# VEA V6 Architecture Guide (纯标签信息流与预设边拼凑架构指南)

> **“图即 `{contents: [], edges: []}`；只加处理边，无输入则由预设边从现有信息找两条拼凑；外部检测 `tag=result` 终止。”**

This document describes **VEA V6** (`vea_v6/`), the pure tag-driven information-flow
evolution of the Vertex-Edge Agent framework. It coexists cleanly alongside
`vea_v5/` ([ARCHITECTURE_V5.md](ARCHITECTURE_V5.md)) and `framework/`
([ARCHITECTURE.md](ARCHITECTURE.md)).

---

## 1. Core design of V6 (核心设计理念)

VEA V6 removes named vertex objects altogether and represents a task DAG purely as
`{"contents": [], "edges": []}`:

| Concept | V6 |
| :--- | :--- | :--- |
| Graph state | `{"contents": [], "edges": []}` (`dag.contents`, `dag.edges`) |
| Adding computation | `dag.add_edge(input_tags, processor, output_tags)` |
| Processor format | `prompt: str` \| `{"模型": str, "处理prompt": str}` (or `Callable`) |
| Data unit | `Vertex(name, content, status)` | `{"tags": list[str], "content": str, "image"?: path\|url, "gen": 0}` |
| Missing-input / multi-branch merge | Built-in `dag.preset_edge` picks 2 items from `contents` and pieces them into 1 item |
| Execution & termination | `dag.run()` runs once, `dag.context` lists all items, external loop checks `tag="result"` |

---

## 2. Information & Edge primitives (`vea_v6/primitives.py`)

### 2.1 Information entry (`ContentItem`)

Every message fed into or produced by a V6 DAG has the shape:

```python
{
    "tags": ["input_tag"],          # list[str]
    "content": "message payload",   # str
    "image": "path/or/url.png",     # optional: omitted when not present
    "gen": 0,                       # generation counter (0 for initial inputs)
}
```

`ContentItem` subclasses `dict`, so both `item["tags"]` / `item["content"]` /
`item["gen"]` and `item.tags` / `item.content` / `item.image` / `item.gen` work
interchangeably.

### 2.2 Processing edge (`Edge`)

An edge is a 3-tuple `(input_tags, processor, output_tags)`:

- `input_tags`: `list[str]` — required input tags (may be empty `[]`).
- `processor`:
  - `str` — prompt string (e.g. `"提取核心数字: {content}"`), or
  - `dict` — `{"模型": "sensenova-v6", "处理prompt": "按汇率折算: {content}"}` (also
    accepts English keys `{"model": ..., "prompt": ...}`), or
  - `Callable` — a Python function for deterministic transforms and unit tests.
- `output_tags`: `list[str]` — tags attached to the newly generated message.

---

## 3. Preset edge & two-item synthesis (`dag.preset_edge`)

Every `DAG` instance initializes a preset edge object (`dag.preset_edge`) at
construction time.

When an edge in `dag.edges` has **no direct single input** during `dag.run()` —
either because its `input_tags` is empty (`[]`), or because no single item in
`dag.contents` carries all of its required `input_tags` (for example, when two
upstream branches produced `["price"]` and `["discount"]` separately and the
downstream edge requires `["price", "discount"]`):

1. The DAG searches `dag.contents` and selects the best two existing items
   (prioritizing tag coverage, semantic similarity, unconsumed leaf items, and
   higher `gen`).
2. Those two items are passed through `dag.preset_edge` to piece together a
   single synthesized item (`gen = max(gen1, gen2) + 1`), which is appended to
   `dag.contents`.
3. The edge then runs its `processor` on that synthesized item and appends its
   output (`gen = synthesized.gen + 1`) to `dag.contents`.

---

## 4. Running a task & external termination (`dag.run()` & `dag.context`)

A `DAG` represents a single task:

```python
from vea_v6 import DAG

dag = DAG()
dag.add_edge(["query"], "拆解问题: {content}", ["sub_plan"])
dag.add_edge(["sub_plan"], {"模型": "gpt-4o", "处理prompt": "生成最终解答: {content}"}, ["result"])

dag.send({"tags": ["query"], "content": "如何设计极简智能体框架？", "gen": 0})

# External termination check on tag="result"
while not any("result" in item["tags"] for item in dag.context):
    dag.run()

print(dag.get_result())
```

---

## 5. The DAG that builds a DAG (`vea_v6/builder.py`)

Just as V6 ships with a `SentenceToDAGMerger`-style tool, it also ships with
`create_dag_builder_dag()` and `MetaDAGBuilder`:

- `builder_dag = create_dag_builder_dag()` is itself a genuine 3-edge V6 `DAG`:
  1. `(["input"], _edge1_parse_edges, ["parsed_edges"])` — extracts V6 edges from
     natural language sentences or edge specs.
  2. `(["input"], _edge2_analyze_alignment, ["tag_alignment"])` — aligns similar
     upstream/downstream tags and decomposes composite multi-branch tags.
  3. `([], _edge3_compile_target_dag, ["result"])` — has no single input, so it
     uses `builder_dag.preset_edge` to piece together `parsed_edges` and
     `tag_alignment`, then compiles the target V6 `DAG` (`{"contents": [], "edges": [...]}`)
     into a message with `tags=["result"]`.

---

## 6. Verified commands (实测命令)

```bash
python3 examples/05_vea_v6_quickstart.py
# 最终回应 (tag=result): {"net_profit": 2000, "status": "盈利"}

python3 examples/06_vea_v6_meta_builder.py
# 【步骤 3】构建出的业务 DAG 运行完毕，最终回应 (tag=result): {"final_pay": 220}

python3 tests/test_vea_v6_all.py
# ALL 7 VEA V6 TEST SUITES PASSED SUCCESSFULLY!
```
