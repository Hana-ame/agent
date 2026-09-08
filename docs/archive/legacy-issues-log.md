# 🛠️ 历史架构问题解决记录归档 (Legacy Resolved Architectural Issues)

本文档归档了 Vertex-Edge Agent Framework 早期迭代过程中排查并解决的 11 项历史架构问题记录。

---

## 1. Issue 1: Legacy top-level `prompt/model/agent` fields rejected

### Problem
Legacy schemas allowed specifying `prompt`, `model`, `threshold`, and `pipeline` at the top level of edges. New validations in `_reject_legacy_keys()` raise `ValueError` to enforce migration.

### Solution
Consolidate compute execution parameters inside `settings`: `settings.prompt / settings.model / settings.threshold / settings.operator`. Only structural routing keys remain at the top level (`id / source / destination / channel / max_iterations / script`).

### Changes
- `framework/graph.py`: Added deprecation enforcement for legacy keys.
- Configuration examples: Updated JSON configurations to place execution attributes inside `settings`.

### Verification
- Top-level legacy keys raise `ValueError` as expected.
- Attributes placed inside `settings` are consumed correctly across the test suite.

---

## 2. Issue 2: Documentation referenced non-existent `ProxiedLLMAgent`

### Problem
Documentation formerly referenced a `ProxiedLLMAgent`, but no such class exists in the codebase (`get_agent({"type": "proxy"})` raised `ValueError: Unsupported agent config type: proxy`).

### Solution
Removed all references to `ProxiedLLMAgent` and non-existent configuration fields. Streamlined proxy documentation to reflect actual proxy capabilities in `HttpLLMAgent` and `OpenCodeAgent`.

### Changes
- Removed fictitious proxy agent entries and references from documentation.

### Verification
- Repository audit verifies zero remaining references to `ProxiedLLMAgent`.

---

## 3. Issue 3: `script` key misdescribed as injecting top-level hooks

### Problem
Earlier documentation claimed `script` was used to inject top-level functions like `on_receive` or `pre_process`. In reality, the framework instantiates subclasses of `Vertex`, `Edge`, or `MapEdge`.

### Solution
Clarified documentation: `script` takes `file.py:ClassName` and instantiates subclasses overriding lifecycle methods.

### Changes
- Updated configuration guidelines and script loader docstrings to specify class-based inheritance.

### Verification
- All example scripts inherit from `Vertex` or `Edge` and pass regression tests.

---

## 4. Issue 4: `GraphBuilder.edge()` retained obsolete `agent` parameter

### Problem
`GraphBuilder.edge()` accepted an `agent` parameter and wrote it to `settings["agent"]`, but `Edge.__init__` does not read this field (agents are owned by custom Edge subclasses or injected by the Executor).

### Solution
Removed the legacy `agent` argument from `GraphBuilder.edge()`.

### Changes
- Removed `agent` parameter from `framework/builders/builder.py`.

### Verification
- Builder clean, all tests pass.

---

## 5. Issue 5: Prompt accumulation during Edge retries

### Problem
When `retry_policy` executed, feedback modified `self.prompt` in-place, accumulating multiple `[SYSTEM FEEDBACK]` blocks across iterations and inflating context tokens.

### Solution
Freeze initial prompt as `self._base_prompt`; rebuild `active_prompt` on each attempt and restore original prompt on completion.

### Changes
- `framework/edge.py`: Store immutable `_base_prompt` and isolate retry prompt state.
- `tests/test_retry_and_stream.py`: Added regression test asserting only a single feedback block is injected.

### Verification
- Regression test verifies prompt does not stack over repeated cycles.

---

## 6. Issue 6: `HttpLLMAgent` connection leak and non-retryable error loops

### Problem
Underlying HTTP clients were not closed explicitly, and non-retryable status codes (400, 401, 403) triggered retries, exhausting rate limits needlessly.

### Solution
Raise `NonRetryableHTTPError` for client errors (400/401/403/404); retry only transient errors (429/500/502/503/504). Added async context manager support and idempotent `close()`.

### Changes
- `framework/agents/_http_base.py`: Added `NonRetryableHTTPError` and client cleanup logic.
- `tests/test_agents.py`: Added tests for clean exit, error categorization, and idempotent close.

### Verification
- Non-retryable errors fail immediately; transient errors retry; resources cleaned up cleanly.

---

## 7. Issue 7: `GraphBuilder.vertex()` stored script in wrong attribute key

### Problem
The builder previously stored scripts under `vc["pipeline"]` while `from_dict()` looked for `vc["script"]`, causing custom vertex classes to be dropped silently.

### Solution
Corrected key assignment to `vc["script"] = script`.

### Changes
- `framework/builders/builder.py`: Storing scripts under `vc["script"]`.

### Verification
- Tests verify custom vertex classes built via `GraphBuilder` load properly.

---

## 8. Issue 8: MapEdge step script path resolution relative to CWD

### Problem
When executed from outside the example directory, relative script paths in pipeline steps failed to resolve.

### Solution
Normalize step script paths relative to the directory of the loaded configuration file.

### Changes
- `framework/graph.py`: Resolve pipeline step scripts relative to config directory.

### Verification
- Demos execute successfully regardless of invocation working directory.

---

## 9. Issue 9: `load_class_from_script` selected subclasses by alphabetical order

### Problem
Auto-discovery selected classes alphabetically, occasionally picking the wrong subclass when multiple classes existed in a file.

### Solution
Prioritize explicit `ClassName` specified after the colon (`script.py:ClassName`) before falling back to auto-discovery.

### Changes
- `framework/utils/script_loader.py`: Lookup requested class by name first.

### Verification
- Regression tests verify precise class loading by name.

---

## 10. Issue 10: Forum crawler timestamp and empty element parsing

### Problem
Forum thread pages with relative timestamp headers and placeholder rating containers failed datetime parsing, causing replies to be filtered out.

### Solution
Refined CSS selector to match exact post containers, stripped locale prefixes before parsing timestamps, and sorted extracted replies chronologically.

### Changes
- `examples/s1_ai_report_map/s1_edges.py`, `tests/fixtures/s1_thread.html`, `tests/test_s1_edges.py`.

### Verification
- Deterministic offline tests against saved fixture parse posts and timestamps correctly.

---

## 11. Issue 11: Accurate test and example counts

### Problem
Documentation formerly cited outdated test and example counts.

### Solution
Synchronized documentation with real test suite metrics.
