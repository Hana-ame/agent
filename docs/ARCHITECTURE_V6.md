# VEA V6 Architecture Guide (纯标签信息流与预设边拼凑架构指南)

> **“图即 `{contents: [], edges: []}`；只加处理边（必有输入/输出 tag，按 layer 从小到大跑）；找不到对应 tag 时由预设边从相近信息合成「输出 tag = 所需输入 tag」的条目；外部检测 `tag=result` 终止。”**

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
| Adding computation | `dag.add_edge(input_tags, processor, output_tags, layer=0)` |
| Processor format | `prompt: str` \| `{"模型": str, "处理prompt": str}` (or `Callable`) |
| Edge ordering | Every edge carries an int `layer`; `dag.run()` always runs edges from the smallest `layer` up (definition order breaks ties) |
| Model registry | `DAG(models={...})`: model name → `{"api", "key", "model", "params"}`; an edge's `{"模型": KEY}` is resolved against this registry at LLM call time (unknown KEY → `KeyError`) |
| Data unit | `{"tags": list[str], "content": str, "image"?: path\|url, "gen": 0}` |
| Missing-tag synthesis | `dag.preset_edge` (default constant `DEFAULT_PRESET_EDGE`) picks 1-or-more similar items from `contents` and synthesizes a new item whose **output tag = the edge's required input tags** |
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

An edge is a 4-tuple `(input_tags, processor, output_tags, layer)`:

- `input_tags`: `list[str]` — required input tags (**must not be empty**).
- `processor`:
  - `str` — prompt string (e.g. `"提取核心数字: {content}"`), or
  - `dict` — `{"模型": "sensenova-v6", "处理prompt": "按汇率折算: {content}"}` (also
    accepts English keys `{"model": ..., "prompt": ...}`); the `"模型"` value is a KEY
    resolved against `dag.models` (api / key / model / params) at LLM call time, or
  - `Callable` — a Python function for deterministic transforms and unit tests.
- `output_tags`: `list[str]` — tags attached to the newly generated message (**must not be empty**).
- `layer`: `int` (default `0`) — execution level; `dag.run()` always runs edges from the
  smallest `layer` up, breaking ties by definition order.

---

## 3. Preset edge & missing-tag synthesis (`dag.preset_edge`)

The default preset edge is the module-level constant `DEFAULT_PRESET_EDGE`
(`vea_v6.dag.DEFAULT_PRESET_EDGE`). Pass it explicitly when constructing a DAG:
`DAG(preset_edge=DEFAULT_PRESET_EDGE, ...)`; it is then accessible as
`dag.preset_edge` (alias `dag.default_edge`).

Processing edges always run from the smallest `layer` up. When `dag.run()`
processes an edge and **finds no item carrying its required input tags** (for
example, when two upstream branches produced `["price"]` and `["discount"]`
separately and the downstream edge requires `["price", "discount"]`):

1. If some not-yet-run upstream edge can still produce those tags directly, the
   current edge is deferred to a later pass.
2. Otherwise the DAG looks at what tags the edge needs, searches `dag.contents`
   for the **most similar one or more items** (prioritizing tag coverage,
   semantic similarity, unconsumed leaf items, and higher `gen`), and passes
   them through `dag.preset_edge`.
3. The preset edge synthesizes a new item whose **output tag = the edge's
   required input tags** (`gen = max(gens) + 1`), appended to `dag.contents`.
4. The edge then runs its `processor` on that synthesized item and appends its
   output (`gen = synthesized.gen + 1`) to `dag.contents`.

---

## 4. Running a task & external termination (`dag.run()` & `dag.context`)

A `DAG` represents a single task:

```python
from vea_v6 import DEFAULT_PRESET_EDGE, DAG

dag = DAG(preset_edge=DEFAULT_PRESET_EDGE)
dag.add_edge(["query"], "拆解问题: {content}", ["sub_plan"], layer=1)
dag.add_edge(["sub_plan"], {"模型": "gpt-4o", "处理prompt": "生成最终解答: {content}"}, ["result"], layer=2)

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
  1. `(["input"], _edge1_parse_edges, ["parsed_edges"], layer=1)` — extracts V6 edges from
     natural language sentences or edge specs.
  2. `(["input"], _edge2_analyze_alignment, ["tag_alignment"], layer=1)` — aligns similar
     upstream/downstream tags and decomposes composite multi-branch tags.
  3. `(["parsed_edges", "tag_alignment"], _edge3_compile_target_dag, ["result"], layer=2)` —
     no single item carries both tags, so `builder_dag.preset_edge` synthesizes a
     `["parsed_edges", "tag_alignment"]` item from the two upstream outputs, then the edge
     compiles the target V6 `DAG` (`{"contents": [], "edges": [...]}`)
     into a message with `tags=["result"]`.

---

## 6. Bootstrap: the DAG that generates a DAG framework (`vea_v6/bootstrap.py`)

`create_bootstrap_dag()` is again a plain V6 `DAG`. Feed it a description, and it delivers a
**complete, runnable, single-file framework** (`minivea`, ~70 KB, zero dependencies) under
`tag="result"`.

| layer | input_tags | output | processor |
|---|---|---|---|
| 1 | `["input"]` | `["framework_spec"]` | description → normalized SPEC (deterministic, or LLM when requested) |
| 2 | `["framework_spec"]` | `["framework_code"]` | SPEC → rendered framework source |
| 3 | `["framework_code"]` | `["fixed_point"]` | regenerate from the code and compare |
| 4 | `["framework_code", "fixed_point"]` | `["result"]` | merge both branches and deliver |

Layer 4 needs **two** input tags, which no single item satisfies — so the preset edge
synthesizes the merged item. That is the real-world use of preset-edge composition, not a toy.

### Why the result is stable (quine fixed point)

The generated file carries two pieces of meta-information in its tail:

- `BODY_B64` — the base64 of the file **without** the tail;
- `SPEC_SPAN` — where the SPEC's base64 sits inside that decoded body.

`generate_framework(spec)` decodes the body, swaps the SPEC span, and re-emits a tail in the
same format. The tail length depends only on the body length, and the body length only on the
SPEC length — **the tail never depends on itself**, so there is no self-referential equation
and no iteration to converge. Consequences:

1. Same SPEC ⇒ byte-identical source, always (`verify_fixed_point`);
2. Load the generated file as a module and call *its own* `generate_framework()` ⇒ it rebuilds
   itself byte-for-byte (`verify_self_reproduction`) — that is the bootstrap;
3. Change the SPEC (same prompt, different description) ⇒ **a different framework**, which is
   itself stable and self-reproducing.

`check_tail_format_sync()` guards the one fragile spot: the tail format string exists twice
(once in the module, once inside the produced code). If the two drift apart, the fixed point
would break silently, so the renderer compares them and raises instead.

> Note: a naive design that stores the whole file inside itself does not work — the length
> equation `L = C + 2 + 4*ceil(L/3)` has no fixed point (the coefficient is 4/3, not 1), so
> iteration diverges monotonically. Excluding the tail from what gets embedded is what makes
> the system closed and solvable.

---

## 7. Verified commands (实测命令)

```bash
python3 examples/05_vea_v6_quickstart.py
# 最终回应 (tag=result): {"net_profit": 2000, "status": "盈利"}

python3 examples/06_vea_v6_meta_builder.py
# 【步骤 3】构建出的业务 DAG 运行完毕，最终回应 (tag=result): {"final_pay": 220}

python3 examples/07_vea_v6_tag_reverse_openai.py
python3 examples/08_vea_v6_openai_edge_configs.py
# 注册表 / 内联 / 混合 三种 OpenAI 兼容边配置解析全部 ✓

python3 examples/09_vea_v6_bootstrap.py
# 自举产出 72385 字节；固定点 ✓；生成物自身复现 ✓；换描述生成新框架 ✓

python3 tests/test_vea_v6_all.py
# ALL 18 VEA V6 TEST SUITES PASSED SUCCESSFULLY!
```
