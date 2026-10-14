# VEA V5 API & Tools Guide (API 与工具指南)

Reference for the `vea_v5` package: public classes, their signatures, and the
six tool-call primitives. Architecture and rationale live in
[ARCHITECTURE_V5.md](ARCHITECTURE_V5.md); the quick start is README §9.

Every signature below was checked against the source with `inspect.signature`
or the dataclass field list — see `vea_v5/*.py`.

---

## 1. Package layout (包结构)

| Module | Contents |
| :--- | :--- |
| [`vea_v5/primitives.py`](../vea_v5/primitives.py) | `Vertex`, `VertexStatus`, `Edge`, `VirtualEdge`, `Triplet`, `MergeMode` |
| [`vea_v5/dag.py`](../vea_v5/dag.py) | `DAG`, `CycleError`, `default_text_similarity` |
| [`vea_v5/executor.py`](../vea_v5/executor.py) | `Executor`, `ExecutionReport`, `EdgeTrace`, `merge_values` |
| [`vea_v5/llm.py`](../vea_v5/llm.py) | `LLMProtocol`, `OpenAICompatibleLLM`, `SmartMockLLM` |
| [`vea_v5/merger.py`](../vea_v5/merger.py) | `SentenceToDAGMerger`, `MergerResult`, `parse_vev_from_text` |
| [`vea_v5/toolkit.py`](../vea_v5/toolkit.py) | `DAGToolKit`, `VEA_V5_TOOLS_SCHEMA` |
| [`vea_v5/closed_loop.py`](../vea_v5/closed_loop.py) | `ClosedLoopRunner`, `VerifyResult`, `ClosedLoopSummary`, `LoopIterationRecord` |

`vea_v5/__init__.py` re-exports all of the above plus `__version__ == "5.0.0"`.

---

## 2. Primitives (`vea_v5.primitives`)

### `VertexStatus(str, Enum)`
`PENDING = "pending"` · `READY = "ready"` · `FAILED = "failed"`

### `MergeMode(str, Enum)`
`OVERWRITE = "overwrite"` · `JSON_MERGE = "json_merge"` ·
`LIST_APPEND = "list_append"` · `TEXT_JOIN = "text_join"`

### `Vertex` (dataclass)
```python
Vertex(name: str, content: Any = None, description: str = "",
       status: VertexStatus = VertexStatus.PENDING,
       metadata: Dict[str, Any] = field(default_factory=dict))
```
Properties / methods:
- `is_ready -> bool` — `status == READY`
- `semantic_text -> str` — `"name: description"` or just `name`
- `set_ready(value)` — write content, status → `READY`
- `reset()` — clear content, status → `PENDING`
- `content_as_str() -> str` — JSON-serialized (or `str()`) for LLM rendering
- `to_dict() -> Dict`

### `Edge` (dataclass)
```python
Edge(id: str, source: str, target: str, action: str = "", kind: str = "llm",
     prompt: Optional[str] = None, fn: Optional[Callable[[Any], Any]] = None,
     code: Optional[str] = None, merge_mode: MergeMode = MergeMode.OVERWRITE,
     metadata: Dict[str, Any] = field(default_factory=dict))
```
- `kind` — `"llm"` | `"code"` | `"passthrough"` | `"virtual"`
- `render_prompt(input_text) -> str` — substitutes `{input}`; otherwise appends
  `输入数据：\n{input_text}` to the instruction
- `is_virtual -> bool` — always `False` on `Edge`
- `to_dict() -> Dict`

### `VirtualEdge(Edge)` (dataclass)
```python
VirtualEdge(id: str, source: str, target: str, action: str = "",
            merge_mode: MergeMode = MergeMode.TEXT_JOIN, similarity: float = 1.0,
            reason: str = "语义相似自动连接")
```
`__post_init__` forces `kind = "virtual"` and upgrades an `OVERWRITE`
`merge_mode` to `TEXT_JOIN`. `is_virtual -> True`.

### `Triplet` (dataclass)
```python
Triplet(source_name: str, edge_action: str, target_name: str,
        source_desc: str = "", target_desc: str = "", edge_kind: str = "llm",
        prompt: Optional[str] = None, code: Optional[str] = None,
        fn: Optional[Callable[[Any], Any]] = None, raw_sentence: str = "")
```

---

## 3. `DAG` (`vea_v5.dag`)

```python
DAG(name: str = "vea_v5_dag")
```

Construction:
- `add_vertex(name, content=None, description="", metadata=None) -> Vertex` —
  insert or patch; new content invalidates downstream
- `add_edge(source, target, action="", edge_id=None, kind=None, prompt=None,
  fn=None, code=None, merge_mode=MergeMode.OVERWRITE, metadata=None) -> Edge` —
  `fn`/`code` force `kind="code"`; `edge_id` defaults to
  `e_{n}_{source}_to_{target}`; cycle → rollback + `CycleError`
- `add_virtual_edge(source, target, similarity=1.0,
  reason="相似顶点自动桥接", merge_mode=MergeMode.TEXT_JOIN, edge_id=None) -> VirtualEdge`
- `add_step(source, action, target, *, edge_id=None, prompt=None, code=None,
  merge_mode=MergeMode.OVERWRITE, source_desc="", target_desc="") -> Edge` —
  callable `action` → `code` edge; string `action` → `llm` edge
- `add_triplet(triplet: Triplet) -> Edge`
- `remove_edge(edge_id) -> bool`
- `reconnect_edge(edge_id, new_source=None, new_target=None) -> Edge`

Analysis / mutation:
- `in_degrees() / out_degrees() -> Dict[str, int]`
- `incoming_edges(name) / outgoing_edges(name) -> List[Edge]`
- `roots() -> List[str]` — in-degree 0
- `sinks() -> List[str]` — out-degree 0 and in-degree > 0 (fallback: all vertices)
- `get_downstream_vertices(start) -> List[str]` — BFS order
- `invalidate_downstream(v) -> List[str]` — reset descendants only
- `invalidate_from(v) -> List[str]` — reset `v` + descendants
- `find_cycle() -> Optional[List[str]]` — three-colour DFS
- `topological_generations() -> List[List[str]]` — Kahn tiers; raises `CycleError`
- `auto_link_similar_vertices(threshold=0.45,
  similarity_fn=default_text_similarity, merge_mode=MergeMode.TEXT_JOIN) -> List[VirtualEdge]`
- `visualize_ascii() -> str` · `to_dict() -> Dict`

### `default_text_similarity(text_a, text_b) -> float`
Combined semantic similarity (0.0–1.0): normalized edit/substring containment,
1-gram + 2-gram character cosine, token Jaccard.

### `CycleError(ValueError)`
Raised (with the cycle path) whenever an edge insertion/reconnection would form
a cycle; the offending edge is rolled back first.

---

## 4. `Executor` (`vea_v5.executor`)

```python
Executor(dag: DAG, llm: Optional[LLMProtocol] = None, max_concurrency: int = 8)
await executor.run(inputs: Optional[Dict[str, Any]] = None) -> ExecutionReport
executor.run_sync(inputs: Optional[Dict[str, Any]] = None) -> ExecutionReport
```

### `ExecutionReport` (dataclass)
`success: bool` · `executed_edges: List[str]` · `skipped_vertices: List[str]` ·
`outputs: Dict[str, Any]` (sink contents) · `all_vertices: Dict[str, Any]` ·
`traces: List[EdgeTrace]` · `errors: List[str]` · `elapsed_ms: float`
- `final_output -> Any` — unwraps a single sink; dict when several

### `EdgeTrace` (dataclass)
`edge_id` · `source` · `target` · `kind` · `action` · `is_virtual` ·
`input_value` · `output_value` · `elapsed_ms` · `success` · `error: Optional[str]`

### `merge_values(existing, incoming, mode: MergeMode) -> Any`
Settlement per `MergeMode`: `overwrite` replaces; `text_join` joins strings
(dicts merge); `list_append` collects into a list; `json_merge` shallow-merges
dicts (parsing JSON strings when needed).

---

## 5. LLM backends (`vea_v5.llm`)

```python
@runtime_checkable
class LLMProtocol(Protocol):
    async def ask(self, prompt: str) -> str: ...

OpenAICompatibleLLM(model: str = "sensenova-6.8-flash-lite",
                    api_key: Optional[str] = None, base_url: Optional[str] = None,
                    temperature: float = 0.2, timeout: float = 30.0)
SmartMockLLM(rules: Optional[Dict[str, Union[str, Callable[[str], Any]]]] = None,
             fallback_fn: Optional[Callable[[str], str]] = None)
```

- `OpenAICompatibleLLM.ask` POSTs `/chat/completions` (via `httpx`, imported
  lazily) to `base_url` (env `VEA_BASE_URL` / `OPENAI_BASE_URL`, defaults to
  OpenAI) with `Authorization: Bearer` from `api_key` (env `VEA_API_KEY` /
  `OPENAI_API_KEY`).
- `SmartMockLLM.ask` matches the longest keyword in `rules`; the handler may be
  a string (returned verbatim) or a callable (returned as JSON if not a str).
  Falls back to `fallback_fn`, then to a default “处理结果” echo. Records every
  prompt in `call_history`.
- `add_rule(keyword, handler)` registers one rule.

---

## 6. Merger (`vea_v5.merger`)

```python
parse_vev_from_text(text: Union[str, Sequence[str]]) -> List[Triplet]

SentenceToDAGMerger(similarity_threshold: float = 0.45,
                    similarity_fn: Callable[[str, str], float] = default_text_similarity,
                    default_merge_mode: MergeMode = MergeMode.TEXT_JOIN,
                    edge_bindings: Optional[Dict[str, Union[Callable[[Any], Any], str]]] = None)
merger.meta_dag -> DAG                 # the 3-layer merger DAG itself
merger.merge(sentences) -> MergerResult
```

`parse_vev_from_text` accepts “从 X 经过/通过/经由/利用/使用/执行/进行/做 Y 到/至/得到/生成/产出/输出 Z”, `X --[Y]--> Z`, `X -> Y -> Z`, numbered lines and
chained clauses. Unparseable lines raise `ValueError` with a hint.

`MergerResult` (dataclass): `compiled_dag: DAG` · `extracted_triplets` ·
`virtual_edges: List[VirtualEdge]` · `similarity_logs: List[str]` ·
`meta_dag_trace: Optional[ExecutionReport]`

`edge_bindings` maps an action string to either a Python callable (inserted as a
`code` edge) or a prompt string (inserted as an `llm` edge).

---

## 7. Toolkit (`vea_v5.toolkit`)

```python
DAGToolKit(dag: Optional[DAG] = None, llm: Optional[LLMProtocol] = None,
           verifier_fn: Optional[Callable[[Any, DAG], Dict[str, Any]]] = None)
await toolkit.call_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]
```

The six primitives (also declared in `VEA_V5_TOOLS_SCHEMA`, OpenAI
`tool_calls` format — see §8):

| Tool | Signature (all return `Dict[str, Any]`) |
| :--- | :--- |
| `add_vev_step` | `(source, action, target, edge_id=None, prompt=None, code=None, merge_mode="overwrite")` |
| `set_vertex_input` | `(name, content, description="")` |
| `link_similar_vertices` | `(source=None, target=None, similarity=0.85, reason="这两个顶点语义相似", threshold=0.45)` — both given → one `VirtualEdge`; neither → whole-graph auto-link |
| `update_edge` | `(edge_id, action=None, prompt=None, code=None, fn=None, merge_mode=None)` — `prompt` ⇒ `kind="llm"`; `code`/`fn` ⇒ `kind="code"`; returns the invalidated vertices |
| `remove_edge` | `(edge_id)` |
| `run_and_verify` | `()` — async; runs the DAG and applies `verifier_fn` |

`run_and_verify` returns `execution_success`, `executed_edges`,
`skipped_cached_vertices`, `final_output`, `all_vertices`, `errors`, and
`verification` (the `verifier_fn` result, or `{"passed": report.success}` when
no verifier is set).

---

## 8. `VEA_V5_TOOLS_SCHEMA` (工具 Schema)

A JSON list of 6 OpenAI function definitions, one per primitive above. Shape of
each entry:

```json
{
  "type": "function",
  "function": {
    "name": "add_vev_step",
    "description": "向 DAG 中添加一步 '从 source 经过 action 到 target' 的 V-E-V 计算步骤。若顶点不存在会自动创建。",
    "parameters": {
      "type": "object",
      "properties": {
        "source": {"type": "string"},
        "action": {"type": "string"},
        "target": {"type": "string"},
        "edge_id": {"type": "string"},
        "prompt": {"type": "string"},
        "code": {"type": "string"},
        "merge_mode": {"type": "string", "enum": ["overwrite", "text_join", "json_merge", "list_append"]}
      },
      "required": ["source", "action", "target"]
    }
  }
}
```

Pass the whole list to any model that supports `tool_calls`, then dispatch the
model's calls with `await toolkit.call_tool(name, arguments)` — see
`examples/04_llm_toolcall_builder.py`.

---

## 9. Closed loop (`vea_v5.closed_loop`)

```python
ClosedLoopRunner(verifier_fn: Callable[[Any, DAG], Union[VerifyResult, Dict[str, Any]]],
                 optimizer_fn: Optional[Callable[[DAGToolKit, VerifyResult, Dict[str, Any]], List[Dict[str, Any]]]] = None,
                 llm: Optional[LLMProtocol] = None,
                 max_rounds: int = 5,
                 similarity_threshold: float = 0.45)

await runner.run_on_dag(dag, initial_inputs=None) -> ClosedLoopSummary
await runner.run_from_sentences(sentences, initial_inputs, edge_bindings=None) -> ClosedLoopSummary
runner.run_on_dag_sync(dag, initial_inputs=None) -> ClosedLoopSummary
runner.run_from_sentences_sync(sentences, initial_inputs, edge_bindings=None) -> ClosedLoopSummary
```

- `verifier_fn(output, dag)` returns a `VerifyResult` (or a dict with
  `passed`/`score`/`feedback`/`suggestion`, or a plain `bool`).
- When verification fails: `optimizer_fn(toolkit, verify_result, run_report)`
  runs if provided; otherwise `suggestion["tool_calls"]` are executed through
  `toolkit.call_tool`. Zero mutations ends the loop early (no idle spinning).

### Dataclasses
```python
VerifyResult(passed: bool, score: float = 0.0, feedback: str = "",
             suggestion: Optional[Dict[str, Any]] = None)
LoopIterationRecord(round_index: int, executed_edges: List[str],
                    skipped_cached_vertices: List[str], output: Any,
                    verification: VerifyResult,
                    mutations_applied: List[Dict[str, Any]] = field(default_factory=list),
                    dag_ascii_snapshot: str = "")
ClosedLoopSummary(converged: bool, total_rounds: int, final_output: Any,
                  final_dag: DAG, merger_result: Optional[MergerResult] = None,
                  history: List[LoopIterationRecord] = field(default_factory=list))
```

---

## 10. Verified examples & tests (实测示例与测试)

| Command | Expected tail output |
| :--- | :--- |
| `python3 examples/01_hello_vev.py` | `最终终点 z2 的值: {'count': 4, 'sum': 100}` |
| `python3 examples/02_sentence_merger_dag.py` | `最终终点 z2 输出: {'final_pay': 220}` |
| `python3 examples/03_closed_loop_evolution.py` | `闭环最终是否收敛通过: True (共经历 2 轮)` |
| `python3 examples/04_llm_toolcall_builder.py` | `'final_output': 544, 'verification': {'passed': True, 'score': 1.0}` |
| `python3 tests/test_vea_v5_all.py` | `ALL 4 VEA V5 TEST SUITES PASSED SUCCESSFULLY!` |
| `python3 -m pytest tests/test_vea_v5_all.py -q` | `4 passed` |
