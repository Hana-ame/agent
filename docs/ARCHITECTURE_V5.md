# VEA V5 Architecture Guide (新一代极简框架架构全景)

> **“一句话建图，图即数据；相似的顶点，用虚拟边连起来。”**
> *(One sentence builds a graph; similar vertices are bridged by virtual edges)*

This document describes **VEA V5** (`vea_v5/`), the next-generation minimal
rewrite of the Vertex-Edge Agent framework. It is the architectural companion to
[API_AND_TOOLS_GUIDE.md](API_AND_TOOLS_GUIDE.md) (reference) and README §9
(quick start). The V1/V4 line in `framework/` is untouched and documented in
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Design goals (设计目标)

V5 keeps the VEA conviction — *vertices store data, edges compute, execution is
flow* — and removes everything a DAG does not need:

| Abandoned in V5 | Replaced by |
| :--- | :--- |
| V4's 7-state vertex state machine (`idle` → `todo` → `running` → `data ready`/…) | 2 states: `pending` / `ready` (plus `failed` on error) |
| Manual `attributes=["start", "end"]` bookkeeping | Automatic root / sink derivation from in/out degrees |
| SQLite store sync boilerplate | In-memory vertices, patch-style `add_vertex` updates |
| JSON manifest + loader + version negotiation | Imperative `dag.add_step("x1", fn, "z1")` one sentence at a time |
| HTTP server / dashboard stack | Stdlib-only core (`python3` runs every example and test with zero installs) |

What V5 keeps from the V4 lineage: DAG semantics with cycle detection, tiered
topological scheduling, fan-in merge strategies, and async concurrency — each
reimplemented in a fraction of the code.

Two flagship pieces ship with V5:

1. **The first VEA DAG — `SentenceToDAGMerger`**: loose natural-language
   sentences (“从 x1 经过 y1 到 z1”) are parsed into discrete `V-E-V` triplets,
   similar vertices are scored and bridged with `VirtualEdge`s (“这两个挺像的”),
   and the result compiles into a runnable business DAG.
2. **Verifier-driven closed-loop self-evolution — `ClosedLoopRunner`**:
   run → verify → mutate the DAG through 6 tool-call primitives →
   incrementally re-run only the dirty downstream branch until verification passes.

---

## 2. Core primitives (核心原语)

All primitives live in `vea_v5/primitives.py` and are re-exported from
[`vea_v5/__init__.py`](../vea_v5/__init__.py).

### 2.1 `Vertex` — a pure data container

| Field | Meaning |
| :--- | :--- |
| `name` | Unique vertex name (e.g. `"x1"`, `"原始财报"`, `"营收数据"`) |
| `content` | Current data payload (any type; JSON-serialized for LLM rendering) |
| `description` | Natural-language semantic description, used for similarity matching |
| `status` | `VertexStatus`: `pending` (default) / `ready` / `failed` |
| `metadata` | Arbitrary key/value metadata |

Key methods: `set_ready(value)` writes content and flips to `ready`;
`reset()` clears content back to `pending` (used by incremental invalidation);
`semantic_text` returns `"name: description"` (or just the name) as the input
to similarity scoring.

### 2.2 `Edge` — one computation unit

An edge is the “Y” in “从 X 经过 Y 到 Z”: `source` → `target`. Its `kind`
selects the execution mode:

| `kind` | Execution |
| :--- | :--- |
| `"llm"` | Render `prompt` (or `action`) with `{input}` and call the LLM |
| `"code"` | Call `fn(input)` directly, or `exec` the `code` string and look for `transform` / `execute` / `run` / `process` |
| `"passthrough"` | Copy the source content to the target with no computation |
| `"virtual"` | Same as passthrough but carried by a `VirtualEdge` (see below) |

When multiple edges feed one target, `merge_mode` decides the settlement:
`overwrite` (default), `json_merge` (shallow dict merge), `list_append`,
`text_join`.

### 2.3 `VirtualEdge` — “these two look alike”

`VirtualEdge` is a subclass of `Edge` (`kind="virtual"`) that connects two
*differently named but semantically similar* vertices — e.g. `商品价格列表`
vs `商品价格清单`. It carries `similarity` (0.0–1.0) and `reason`. At runtime
it costs **zero LLM tokens**: the source content is bridged into the target
using the edge's `merge_mode` (default `text_join`; dicts are merged instead of
string-joined).

### 2.4 `Triplet` — the sentence-level building block

`Triplet(source_name, edge_action, target_name, ...)` is one discrete `V-E-V`
corresponding to “从 X 经过 Y 到 Z”. `DAG.add_triplet()` inserts it, carrying
optional `prompt` / `code` / `fn` / descriptions and the raw sentence for
tracing.

---

## 3. `DAG` — the graph container (图容器)

`vea_v5/dag.py`. The container is deliberately small: vertices and edges are
plain dicts, everything else is derived.

### 3.1 Building primitives

- `add_step(source, action, target, ...)` — **the one-sentence primitive**:
  `dag.add_step("x1", "提取关键数字", "z1")`. A callable `action` becomes a
  `code` edge; a string becomes an `llm` edge (its `prompt` defaults to the
  action text).
- `add_vertex(name, content=None, description="", metadata=None)` — insert or
  **patch** an existing vertex (never loses prior data); writing new content
  auto-invalidates downstream.
- `add_edge(source, target, action="", kind=None, prompt=None, fn=None,
  code=None, merge_mode=OVERWRITE, metadata=None)` — the full form behind
  `add_step`.
- `add_virtual_edge(source, target, similarity=1.0, reason=..., merge_mode=TEXT_JOIN)`.
- `remove_edge(edge_id)` / `reconnect_edge(edge_id, new_source=None, new_target=None)`
  — with automatic downstream invalidation and cycle rollback.

### 3.2 Automatic roots / sinks (自动推导起点终点)

- `roots()` — vertices with in-degree 0 (entry points).
- `sinks()` — vertices with out-degree 0 **and** in-degree > 0 (final outputs;
  falls back to all vertices when the graph has no such vertex).

No `attributes=["start", "end"]` annotations are ever needed.

### 3.3 Cycle protection with rollback (成环保护)

`add_edge` / `add_virtual_edge` / `reconnect_edge` run a three-colour DFS
(`find_cycle()`) immediately after insertion. On a cycle, the new edge is
**rolled back** and `CycleError` is raised — the pre-existing DAG is never
polluted (verified by test 2 in `tests/test_vea_v5_all.py`).

### 3.4 Incremental invalidation (增量失效)

- `invalidate_downstream(v)` — reset all descendants of `v` to `pending`
  (keeps `v`'s own data).
- `invalidate_from(v)` — reset `v` and all its descendants.

This is what makes closed-loop re-runs cheap: a mutated edge only dirties its
downstream branch; untouched upstream results stay cached as `ready`.

### 3.5 Similar-vertex auto-linking (相似顶点自动连接)

`auto_link_similar_vertices(threshold=0.45, similarity_fn=default_text_similarity,
merge_mode=TEXT_JOIN)` scans the graph for unfed consumer vertices (in-degree 0,
out-degree > 0, no initial content) and bridges them from the most similar
producer vertices via `VirtualEdge`s, highest similarity first, skipping pairs
that would form a cycle. The default `default_text_similarity` combines
normalized edit/substring containment, character 1-gram + 2-gram cosine (sensitive
to Chinese phrasing such as `商品价格列表` vs `商品价格清单`), and
English/number token Jaccard.

### 3.6 Scheduling & visualization

- `topological_generations()` — Kahn's algorithm; returns tiered layers whose
  vertices can run fully concurrently (raises `CycleError` if cyclic).
- `visualize_ascii()` — prints tiers plus the edge list, distinguishing real
  edges (`===`) from virtual ones (`- -`).

---

## 4. `Executor` — incremental tiered execution (增量拓扑执行器)

`vea_v5/executor.py`. Constructed as `Executor(dag, llm=None, max_concurrency=8)`;
`llm` defaults to a `SmartMockLLM` so any graph runs offline.

Execution loop:

1. Seed `inputs` into their vertices (single-root graphs accept a bare value).
2. Walk `topological_generations()` tier by tier.
3. For each target vertex, run all its incoming edges concurrently under a
   semaphore (`asyncio.gather`), then settle the results in stable edge-id order
   via `merge_values(existing, incoming, merge_mode)` — deterministic despite
   concurrency.
4. **Incremental skip**: a target already `ready` is skipped entirely
   (`skipped_vertices`); a target whose upstream is not ready is marked `failed`
   with a descriptive error.
5. Virtual edges never call the LLM — they bridge the source content directly.

The result is an `ExecutionReport`: `success`, `executed_edges`,
`skipped_vertices`, `outputs` (sink contents), `all_vertices` (every vertex),
`traces` (`EdgeTrace` per edge: input/output/elapsed/success/error),
`errors`, `elapsed_ms`. `final_output` unwraps a single sink, or returns the
sink dict when there are several.

`run(inputs)` is async; `run_sync(inputs)` wraps it (and, when called from
inside a running event loop, offloads to a worker thread instead of refusing).

---

## 5. The first VEA DAG — `SentenceToDAGMerger` (合并器)

`vea_v5/merger.py`. The merger is itself a standard 3-layer V5 DAG
(`meta_dag`), executing end to end under the normal `Executor`:

```
v_raw_sentences
      │  e_1_extract_vev          — parse each sentence into a Triplet
      ▼
v_discrete_vevs
      │  e_2_align_similar_vertices — score producer/consumer pairs, plan VirtualEdges
      ▼
v_aligned_plan
      │  e_3_compile_dag          — assemble real edges + VirtualEdges into a business DAG
      ▼
v_executable_dag
```

- `parse_vev_from_text(text)` handles Chinese “从 X 经过/通过/利用/执行 Y 到 Z”
  forms, `X --[Y]--> Z`, `X -> Y -> Z`, numbered lines, and chains.
- `SentenceToDAGMerger(similarity_threshold=0.45, similarity_fn=..., default_merge_mode=TEXT_JOIN, edge_bindings={action: fn_or_prompt})` —
  `edge_bindings` lets you give concrete Python functions or prompts to the
  actions found in the sentences.
- `merger.merge(sentences) -> MergerResult` returns
  `compiled_dag`, `extracted_triplets`, `virtual_edges`, `similarity_logs`
  (the “这两个挺像的” lines) and `meta_dag_trace`.

Verified with `python3 examples/02_sentence_merger_dag.py` — four loose
sentences about prices/discounts compile into a 7-vertex DAG with three virtual
edges, and `z2` settles to `{'final_pay': 220}`:

```
发现相似顶点: '优惠折扣信息' 与 '原价小计与优惠折扣信息' 挺像的 (相似度=0.74) -> 建立虚拟边 [优惠折扣信息 - -> 原价小计与优惠折扣信息]
发现相似顶点: '商品价格列表' 与 '商品价格清单' 挺像的 (相似度=0.64) -> 建立虚拟边 [商品价格列表 - -> 商品价格清单]
```

---

## 6. Closed-loop self-evolution (闭环自演进)

`vea_v5/closed_loop.py` + `vea_v5/toolkit.py`.

### 6.1 The loop

`ClosedLoopRunner(verifier_fn, optimizer_fn=None, llm=None, max_rounds=5, similarity_threshold=0.45)`:

1. **Run** — execute the current DAG with `DAGToolKit.run_and_verify()`.
2. **Verify** — `verifier_fn(final_output, dag)` returns a `VerifyResult`
   (`passed`, `score`, `feedback`, optional `suggestion`).
3. **Mutate** — if not passed, either the custom `optimizer_fn(toolkit, verify_result, run_report)`
   issues mutations, or the engine auto-executes `suggestion["tool_calls"]`
   through `toolkit.call_tool()` (e.g. `update_edge` / `add_vev_step`).
4. **Re-run incrementally** — only the invalidated downstream branch executes
   again; untouched `ready` vertices are skipped. Repeat until `passed` or
   `max_rounds`.

`run_on_dag(dag, initial_inputs)` and `run_from_sentences(sentences, initial_inputs, edge_bindings)`
(plus their `_sync` twins) return a `ClosedLoopSummary` (`converged`,
`total_rounds`, `final_output`, `final_dag`, `merger_result`, `history` of
`LoopIterationRecord`s — each with executed edges, skipped cached vertices,
output, verification, and applied mutations).

### 6.2 Incremental re-run, proven

`examples/03_closed_loop_evolution.py` starts from a buggy finance DAG
(missing R&D cost, tax not deducted). Round 1 produces `{'net_profit': 3000}`
and fails verification; `optimize_dag_on_failure` updates two edges; round 2
executes only the dirty branch — the untouched `营收数据` vertex is skipped
(reused from cache) — and converges at `{'revenue': 5000, 'cost': 3000, 'net_profit': 1600.0}`:

```
第 2 轮闭环执行: 实际执行的边: ['e_2_原始财报_to_成本数据', 've_4_成本数据_sim_营收数据与成本数据', 've_5_营收数据_sim_营收数据与成本数据', 'e_3_营收数据与成本数据_to_利润报告']
跳过(复用缓存)的顶点: ['原始财报', '营收数据']
闭环最终是否收敛通过: True (共经历 2 轮)
```

The unit test `test_4_incremental_closed_loop_evolution` pins the core
guarantee: the expensive upstream edge `e1` runs exactly once across the whole
loop because round 2 only re-runs `e2`.

---

## 7. `DAGToolKit` — 6 tool-call primitives (Tool-Call 原语)

`vea_v5/toolkit.py` exposes the whole graph API to an LLM as six atomic
function-calling tools (`VEA_V5_TOOLS_SCHEMA`, OpenAI `tool_calls` format):

| # | Tool | Effect |
| :--- | :--- | :--- |
| 1 | `add_vev_step` | Insert `source --action--> target` (auto-creates vertices) |
| 2 | `set_vertex_input` | Write initial content into a vertex (marks `ready`) |
| 3 | `link_similar_vertices` | Insert one `VirtualEdge`, or `threshold`-scan the whole graph |
| 4 | `update_edge` | Rewrite an edge's `prompt` / `code` / `fn`; invalidates downstream |
| 5 | `remove_edge` | Delete an edge; resets its target's downstream |
| 6 | `run_and_verify` | Execute the DAG and run the verifier |

`examples/04_llm_toolcall_builder.py` shows an agent building a graph through
these tools alone: it misspells a vertex (`大数列表` vs `过滤后的大数列表`),
auto-links them via tool 3 (similarity 0.72), fails verification on a missing
square step, fixes `e_sum` via tool 4, and the second run executes **only**
`e_sum` — the filter branch is cached:

```
第二轮增量重跑结果: {'executed_edges': ['e_sum'], 'skipped_cached_vertices': ['x1', '过滤后的大数列表', '大数列表'], 'final_output': 544, ...}
```

---

## 8. LLM backends (模型后端)

`vea_v5/llm.py`:

- `LLMProtocol` — runtime-checkable protocol: implement `async def ask(prompt) -> str`.
- `OpenAICompatibleLLM` — any OpenAI-compatible `/chat/completions` endpoint
  (OpenAI, SenseNova, DeepSeek, Ollama, vLLM). Reads `VEA_API_KEY` /
  `OPENAI_API_KEY` / `VEA_BASE_URL` / `OPENAI_BASE_URL` env vars; requires
  `httpx` (already a project dependency).
- `SmartMockLLM` — offline rule-based mock: `rules` maps keyword → string or
  handler; unmatched prompts fall back to `fallback_fn` or a default
  “processed” echo. Every example and test in this repo runs on it with **no
  API key and no network**.

---

## 9. Verified commands (实测命令)

All commands below were run in this checkout and their outputs pasted:

```bash
python3 examples/01_hello_vev.py
# 执行成功: True, 最终终点 z2 的值: {'count': 4, 'sum': 100}

python3 examples/02_sentence_merger_dag.py
# 最终终点 z2 输出: {'final_pay': 220}

python3 examples/03_closed_loop_evolution.py
# 闭环最终是否收敛通过: True (共经历 2 轮)

python3 examples/04_llm_toolcall_builder.py
# 最终输出: {'execution_success': True, 'final_output': 544, 'verification': {'passed': True, 'score': 1.0}}

python3 tests/test_vea_v5_all.py
# ALL 4 VEA V5 TEST SUITES PASSED SUCCESSFULLY!

python3 -m pytest tests/test_vea_v5_all.py -q
# 4 passed in 0.04s
```

The V5 engine imports **no third-party modules** at runtime: stdlib only
(`asyncio`, `json`, `re`, `math`, `dataclasses`, `enum`, `typing`, `inspect`,
`time`, `concurrent.futures`); `httpx` is imported lazily and only inside
`OpenAICompatibleLLM.ask`.
