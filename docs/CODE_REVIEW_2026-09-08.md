# Code Review — Vertex-Edge Agent Framework (VEA v4)

> **Remediation status (updated)**: the five P0 security findings, the three
> critical engine findings, and H1–H5/H8/H9 have been fixed and covered by
> regression tests; the lifecycle-state/traversal-colour split landed with a SQLite
> migration; and edges can now be extended with **zero registration** via
> `"type": "my_edge.py:MyEdge"`. Full suite: **701 passed, 0 failed** (the V4 runner
> `vea-run-v4` and the FastAPI-free core import, added afterwards, are documented in
> `docs/REDESIGN_RFC.md`).
> A follow-up audit of the *client-facing* extension surface found three further
> defects, all fixed: the dashboard's edge-type `<select>` (4 hardcoded options)
> silently reset `llm_tool`/custom edges to `code`; `create_v4_server(script_roots=...)`
> was stored but never used; and `settings._edge_class_spec` recorded a
> repository-root path that did not exist, so a stored custom edge failed to
> rehydrate. Packaging was also fixed (undeclared `fastapi`/`uvicorn`, dashboard
> assets missing from the wheel), and CI now installs the package on every push
> (`install-and-smoke` in `.github/workflows/ci.yml`) instead of only testing the
> source tree.
> See `docs/REDESIGN_RFC.md` for what changed, migration notes, and the remaining
> structural work (async store, snapshot policy, V4 hooks, OpenAI/SSE contract,
> V1 retirement). Findings below are kept as the original record.

- **Date**: 2026-09-08
- **Revision reviewed**: `8cc16ef` on branch `vertex-edge-agent` (working tree clean)
- **Method**: full static read of the 83 `framework/` modules and 40 test modules; four parallel deep-dive
  reviews (server/API, executors, store/graph, tests/CI); **every critical and high finding below was
  independently reproduced** in an isolated dependency tree (`/tmp/vea-pkgs`, Python 3.12.3) with a
  `PYTHONPATH` override. No repository file was modified during the review.

---

## 0. Verdict

The core idea is sound and the code is unusually well-commented for its size. The problems are not
sloppiness — they are **boundary and semantics problems**:

| Area | Assessment |
| :--- | :--- |
| Architecture / state model | Strong concept, but the lifecycle state enum is polluted with DFS colors and the snapshot design is O(N²) I/O |
| Engine correctness | **Not safe**: the flagship fan-in barrier livelocks, one failed edge permanently disables itself, and the harness loop is unbounded |
| HTTP/API security | **Not safe to expose**: unauthenticated RCE (two independent vectors), SSRF, wildcard CORS |
| Persistence | Correct SQL hygiene (no injection found), but silent data loss in merges, upserts and `llm_tool` serialization |
| Tests | 558 pass / 0 fail — **but they assert the defects as intended behavior** |
| Docs / packaging | Extensive but drifted; `pip install -e .` alone produces an unimportable package |

> **Do not bind this server to `0.0.0.0` until §1 is fixed.** The README currently instructs exactly that.

---

## 1. Critical — security (all reproduced)

### C1. Unauthenticated remote code execution via the edge `script` field
`framework/server/routes/graph.py:161` → `framework/graph_manager_v4.py:180` → `framework/edges/code.py:56-59`

`CodeEdgeV4._resolve_callable()` `eval()`s any string starting with `lambda ` **with `__builtins__`**:

```python
self._callable = eval(stripped, {"__builtins__": __builtins__})
```

The value comes straight from the request body (`EdgeCreateOrUpdateRequest.script`, `schemas.py:42`).
Verified end-to-end against a running app:

```http
POST /api/sessions/s1/graph/edges
{"id":"e_rce","type":"code","input_vertex":"a","output_vertex":"b",
 "script":"lambda content, settings, staging=None: __import__('os').popen('id').read()"}
POST /api/sessions/s1/run
→ vertex b content = "uid=1000(lumin) gid=1000(lumin) groups=1000(lumin),4(adm),..."
```

A second variant uses an absolute file path (`"/tmp/evil.py:execute"`), resolved by
`framework/edges/base.py:87-107` → `framework/utils/script_loader.py:70-75` (`spec.loader.exec_module`).
**Fix**: never `eval`; resolve scripts only inside a configured, allow-listed directory.

### C2. Unauthenticated remote code execution via `vea_manifest_path`
`framework/server/routes/openai.py:221` (read) and `:232-238` (load, no path validation)

`/v1/chat/completions` accepts a client-supplied manifest path and loads it directly. A manifest may
reference an edge `script` (`framework/graphs/loader.py:82-84` → `EdgeV4.from_config`), so this is an
independent RCE vector. Verified: a manifest at `/tmp/evilprobe/manifest.json` referencing
`/tmp/evilprobe/evil.py:execute` was loaded, the file was **executed**, `/tmp/evilprobe/PWNED.txt` was
written, and the response `content` was `"pwned"`.

The same missing check exists in the three subgraph routes:
`routes/graph.py:272`, `:299`, `:327` (`subgraph_manifest`). By contrast the two sites that do validate
(`routes/execution.py:107`, `routes/graph.py:111`) show the intended pattern was simply not applied
everywhere.

### C3. No authentication anywhere, plus wildcard CORS
`framework/server/app.py:50-85` installs only `CORSMiddleware`. There is no `Depends`, API-key, or
middleware check anywhere under `framework/server/`. `app.py:34` defaults `allowed_origins=["*"]` and
`:56-62` allows all methods/headers.

Consequences:
- Every mutation, execution, DB-clear and snapshot-restore endpoint is open to anyone who can reach the port.
- Because CORS is `*` with no credentials required, **any web page the operator visits can drive the
  RCE/SSRF endpoints above from the browser** against `http://127.0.0.1:11434`.
- `README.md:119` and `:589` explicitly document `--host 0.0.0.0`.

**Fix**: API-key dependency on all non-dashboard routers, CORS default `[]`, and a startup warning/refusal
when binding a non-loopback host without a token.

### C4. SSRF and credential passthrough via client-controlled `base_url`
`framework/server/schemas.py:96` → `framework/server/routes/tools.py:78` →
`examples/dynamic_tool_library/tool_scripts.py:87-88`

`POST /api/sessions/{id}/route-and-run` accepts `base_url`, `api_key` and `model` from the caller and
forwards them to an outbound `httpx` POST. Verified with a local listener: the server POSTed the user's
task text plus `Authorization: Bearer CLIENT-SUPPLIED-KEY` to the attacker-chosen URL — an
unauthenticated internal-network request primitive (cloud metadata, internal services) that also
exfiltrates whatever key the caller supplies. **Fix**: drop these fields from the request schema or
validate against a server-side allow-list.

### C5. Path traversal through `session_id` → snapshot read/write outside `snapshot_dir`
`framework/snapshot_v4.py:41-45` (`self.base_dir / session_id`, `mkdir(parents=True)`)

`session_id` is never sanitized. Verified: `POST /api/sessions/%2E%2E/snapshots` returned
`{"path": "/tmp/vea_trav4/base/../step_0000_probe.json"}` and wrote the file **into the parent of the
snapshot directory**; the matching `GET` lists/reads `step_*.json` from there. Attacker-controlled graph
JSON lands on disk and `routes/snapshots.py:55` can restore it as a live graph. The HTTP path-parameter
vector is single-level (`..` is one segment; `%2F` is rejected), but `X-Session-ID` on the OpenAI route
(`routes/openai.py:52`) accepts the value verbatim.
**Fix**: validate `session_id` against `^[A-Za-z0-9._-]{1,64}$` and assert `resolved.is_relative_to(base)`.

---

## 2. Critical — engine correctness (all reproduced)

### E1. Fan-in barrier livelock — data lost, workflow never terminates
`framework/executor_v4.py:347-356` (compute `expected`) vs `:441-457` (accumulate / settle / re-open)

`expected` counts only edges whose `merge_strategy` participates in the barrier, but the completion set
`_fan_in_completed[output]` accepts **any** incoming edge, and the barrier fires when
`len(completed) >= expected`. A non-participating edge that finishes *late* resets the already-settled
vertex back to `todo` (`:457`), which re-enables the earlier edges. Verified with a 3-incoming graph
(e1 overwrite, e2/e3 json_merge → `expected=2`):

```
success: False  elapsed 1.51s  errors: ['Workflow timed out after 1.5s']
runs: e1 e2 e3 e1 e2 e1 e3 ... count: 57
C: todo  content {"e1": 1, "e2": 1}   # e3's contribution permanently lost
```

With the default `timeout=120` this is thousands of redundant edge executions. This is the framework's
headline feature (advertised in `agent.md` §6.1 and the canonical `hn_v4` example).
**Fix**: accumulate only participating edge **ids** and compare sets, not counts.

### E2. A failed edge is permanently blacklisted from scheduling
`framework/executor_v4.py:465-482` (failure path) + `:286` (eligibility filter)

The failure path adds the edge id to `_fan_in_completed[output_vertex]` even when the vertex is not a
fan-in and the barrier never settles — and that set is never cleared. Verified: with `A→C`, `B→C`
(both overwrite) and A's script raising, the run deadlocks (`completed=[]`), the **healthy sibling B is
never dispatched**, and after an external reset of C the edge A is still excluded — permanently, for the
lifetime of the executor instance. **Fix**: only record barrier participants, and clear `_fan_in_*` when
the vertex leaves the settled state.

### E3. Unbounded `while True` in the request-driven harness loop
`framework/http_executor_v4.py:326-537` (`continue` at `:498` and `:536`)

`HttpHarnessExecutorV4.step()` has no iteration cap and no progress check. Any edge that fails while
leaving its output vertex in an eligible state is re-selected immediately. Verified: a self-loop
`CodeEdgeV4("e_selfloop","B","B")` whose script always raises ran **50+ times inside a single HTTP
request**, which never returned. With an LLM-backed edge this is unbounded cost and a hung worker.
**Fix**: cap edges-per-request and treat a no-progress iteration as terminal.

---

## 3. High

### H1. `ExecutorV4` fabricates tool-edge results (the tool never runs)
`framework/executor_v4.py:346-366` → `framework/edges/tool.py:108-120`

Batch execution calls `edge.run()` without `tool_output`, and `ToolEdgeV4.run()` falls back to
`in_v.content`. Verified: `A(data ready) --ToolEdgeV4(bash)--> B` through `ExecutorV4` returns
`success=True`, `completed=['e_tool']`, `B.state='data ready'`, `B.content =` the upstream string — no
tool call was emitted and nothing ran. Reachable in production via `POST /api/sse/execute`.
**Fix**: refuse or skip `ToolEdgeV4`/`LLMToolEdgeV4` in `ExecutorV4`.

### H2. Subgraph failure is reported as parent success
`framework/workflow_executor_v4.py:229` (`sub_res = await sub_executor.run()`) is **never inspected**;
`:252-257` unconditionally sets `DATA_READY` and returns the (possibly empty) output. Confirmed by
reading the code path; a child graph that rejects end-to-end yields `success=True, errors=[]` at the
parent.

### H3. `llm_tool` edges cannot be serialized or restored; their tool catalog is dropped
- `EdgeV4.from_config` (`framework/edges/base.py:335-347`) passes `concurrency_limit`/`concurrency_group`
  to `LLMToolEdgeV4`, whose `__init__` (`framework/edges/tool.py:137-149`) does not accept them.
  Verified: `TypeError: LLMToolEdgeV4.__init__() got an unexpected keyword argument 'concurrency_limit'`.
  This makes `GraphV4.dump()` → `DiscreteGraphLoaderV4.load_from_dict()` fail for any graph containing an
  `llm_tool` edge, i.e. **snapshot restore and time travel break**.
- `LLMToolEdgeV4.__init__` stores `self.tools` but never writes it into `settings`, and `EdgeV4.to_dict`
  (`base.py:166-197`) has no `tools` key. Verified: `tools` is absent from `to_dict()`; `loader.py:248`
  reads `er.settings.get("tools", [])`, which is therefore always `[]`. The tool definitions are silently
  lost on every persist/restore.

### H4. Live in-memory edge settings diverge from SQLite
`framework/graph_manager_v4.py:165-180`

`add_or_update_edge` flattens `settings` to the top level of the config dict, then `EdgeV4.from_config`
rebuilds `settings` from `data["settings"]` — which does not exist. Verified:

```
in-memory edge.settings: {'model':'m','prompt':'P','agent_mode':'chat','concurrency_group':'llm'}
DB edge settings      : {'temperature':0.1,'merge_strategy':'json_merge','model':'m', ...}
```

So `temperature`, `merge_strategy` and any other pass-through key are persisted but **not applied by the
running graph**, and they reappear after a restart — behavior changes across restarts. This is the
divergence the commit `86e016a` claimed to fix.

### H5. The online edge-mutation API only accepts 3 edge types
`framework/graph_manager_v4.py:163-164`

`if edge_type not in ("code","llm","reflexive") and input_vertex != output_vertex: raise`. Verified
rejections: `tool`, `llm_tool`, `llm_chat`, `llm_generate`. `EdgeV4.from_config` supports all of them, and
`agent.md` §2.2 advertises `ToolEdgeV4`/`LLMToolEdgeV4` as core types — but the harness-integration edge
types cannot be configured through the REST API at all.

### H6. Manifest-backed sessions are wiped on every request
`framework/workflow_executor_v4.py:100-107` calls `populate_store` whenever a manifest is supplied, and
`save_vertex` upserts `content`/`state` unconditionally (`framework/vertex_v4.py:409-414`). The OpenAI
route guards against this (`routes/openai.py:238-249`); the workflow executor does not, so a second
request resets completed vertices back to `todo`.

### H7. `derive_session_id` collides unrelated conversations
`framework/http_executor_v4.py:76-85` hashes only the **first** user/system message — normally the shared
system prompt. Two different tasks with the same system prompt map to one session and then share/mutate
one DAG (states, staging, metrics). Used in production at `routes/openai.py:58-59`.

### H8. Early fan-in settle cancels a still-running sibling and reports success
`framework/executor_v4.py:441-457` + `:814-820` + `:782-784`. Verified: e3 is cancelled before its body
runs, yet the result is `success=True, completed_edges=['e1','e2']` and nothing records the loss.

### H9. `except TypeError` re-executes the edge body
`framework/executor_v4.py:357-375` wraps the whole `await edge.run(...)` call in a `try/except TypeError`
used to detect a signature mismatch. A `TypeError` **raised inside the edge body** is misread and the edge
is invoked a second time (without `auto_transition`), while only one metric row is written. All built-in
edges already accept `auto_transition`, so the fallback is dead code for built-ins and pure hazard for
custom edges.

### H10. Merge strategies silently destroy data
`framework/vertex_v4.py:540-549` — `json_merge` overwrites the existing content when the incoming payload
is not valid JSON. `:571-573` — an unknown/misspelled strategy (`"json_merg"`, `"list_apend"`) silently
falls back to `overwrite`. Both verified.

### H11. Partial upserts wipe fields
`framework/vertex_v4.py:411-416` — a content-only `save_vertex` clears `attributes` and resets `state`
(losing `start`/`end` tags, which changes tier computation and eligibility).
`framework/vertex_v4.py:627-636` — a bare `save_edge` blanks `input_vertex`/`output_vertex`/`script`/
`settings` and resets `max_retries`, leaving a persisted dangling edge. Verified.

### H12. Subgraph splice/insert corrupts topology
`framework/graphs/subgraph_ops.py`:
- `:249-250`, `:335-336` — the `_sub` suffix is not re-checked, so an existing edge is silently overwritten.
- `:170-194` — a reflexive edge is rewired into a forward edge (false cycles).
- `:107` + `:197` — a subgraph vertex named like the target deletes that vertex.
- `:130-133` + `:196-197` — an empty subgraph deletes the target and all its incident edges.
- `:226-272` etc. — mutations happen before validation, so a failure leaves a half-inserted graph.
All verified.

### H13. Full graph snapshot on every mutation and every edge completion, with no retention
`framework/graph_manager_v4.py:136/198/205/234` and `framework/executor_v4.py:493-496`. Each write
re-hydrates the whole graph and dumps it to disk: O(N) writes × O(N) size. The repository already holds
**565 snapshot files / ~3.4 MB** produced by the test suite, and there is no prune/retention API.

### H14. SSE and streaming contract defects
- `framework/server/sse.py:125-126` — `finally: yield SSE_DONE` raises
  `RuntimeError: async generator ignored GeneratorExit` on client disconnect (verified).
- `framework/server/sse.py:37-65` — every frame re-emits the complete `tool_calls` object with the same
  `id`/`index` and full `arguments`, so OpenAI clients that accumulate deltas receive concatenated JSON.
- `routes/openai.py:433` — `per_tier` maps to `run_one_tier()` = `step()` = exactly one edge; `per_edge`
  and `per_tier` share a branch, and the streaming branch ignores `execution_mode` entirely.

### H15. `/v1/models` advertises 12 broken models
`framework/server/app.py:88-113` auto-registers **every** `examples/*/config.json` as a V4 template,
including the 12 legacy V1 configs. Verified: `/v1/models` returns 16 ids, and invoking any of the 12
V1-derived ones raises an unhandled `ValueError: invalid literal for int() with base 10: 'input'` → HTTP 500
(`GraphV4.add_vertex` does `int(vertex.get("id", 0))` at `framework/graphs/core.py:133` before checking
`name`).

### H16. `Guard` mode typos fail open
`framework/utils/guard.py:25-28` recognises only the exact strings `"all"`/`"any"`; anything else
(including `"ALL"`) falls through and **ignores the guards**. Verified: a guard that should evaluate
`False` returned `True`.

### H17. `script_loader` pollutes `sys.path` and executes any path
`framework/utils/script_loader.py:57-62` inserts the script's directory at `sys.path[0]` permanently
(module-shadowing / supply-chain vector); `:70-75` executes any path handed to it with no confinement or
allow-list, and never registers the module in `sys.modules`, so repeated loads re-execute the file.

---

## 4. Medium (selected)

| Finding | Location |
| :--- | :--- |
| All SQLite access is synchronous inside `async def` — every vertex write blocks the event loop; no `asyncio.to_thread`/aiosqlite anywhere | `framework/vertex_v4.py:255-272`; call sites in `executor_v4.py`, `http_executor_v4.py`, `server/routes/*` |
| DAG validation writes DFS colors into the persisted lifecycle column; loader then persists them (verified `idle` → `black` in memory, in the store, and in snapshots) | `framework/graphs/validation.py:49-54`, `:82-87` |
| …but note: this does **not** by itself cause the deadlock sometimes attributed to it — `idle` and `black` are both ineligible for the handshake. The real impact is state-fidelity loss and a DB `CHECK` constraint that blocks removing the colors | `framework/vertex_v4.py:288` |
| Traversal colors accepted by the public write API (`save_vertex(..., state="black")` succeeds) | `framework/vertex_v4.py:1019-1027` |
| No schema versioning/migration; `PRAGMA user_version` is 0 and the state `CHECK` is baked into `CREATE TABLE` | `framework/vertex_v4.py:276-380` |
| Unknown model → HTTP 200 with empty content instead of 404 `model_not_found` | `framework/server/routes/openai.py:217,230-262` |
| Valid multimodal `content` (a list) → 500 (`sqlite3.ProgrammingError`); non-dict messages → 500 | `framework/server/routes/openai.py:65-73`, `:152-157` |
| Unbounded per-session memory: `_session_locks`/`_graphs` never evicted; every session id is advertised as a model | `framework/graph_manager_v4.py:55-59,84,112-116` |
| `restore_snapshot` is not transactional (`clear_session` then `populate_store`) and loses `_active_overrides` | `framework/snapshot_v4.py:196-219`; `framework/graphs/core.py:675-701` |
| Topology helpers are O(V·E) and `dump()` calls them twice — 0.24 s for a 400-node graph | `framework/graphs/core.py:484-534,558-655,681-694` |
| Fan-in merge is a non-atomic read-modify-write; the lock is per store instance, so two workers/processes lose updates | `framework/vertex_v4.py:523-575` |
| No retention for `session_staging` / `edge_metrics`; `delete_edge` leaves orphaned metric rows | `framework/vertex_v4.py:670-678,974-987` |
| No request-body size limit; `/api/sse/execute` accepts `max_concurrency=100000` (the sibling route correctly caps at 64) | `framework/server/app.py:50-62`; `routes/execution.py:108-109` |
| No SSE keepalive; `X-Accel-Buffering` set on only one of three streams | `routes/execution.py:77-84`; `server/sse.py:77-126` |
| Slow SSE subscribers silently lose the oldest queued events | `framework/graph_manager_v4.py:481-492` |
| `reenter` in-flight cancellation is unreachable for `/run` (the same session lock is held), so `cancelled_in_flight_edges` is always `[]` | `routes/execution.py:28`; `routes/graph.py:204-222` |
| `GET .../graph/dump?path=…` writes files (state-changing GET) and discards the validated path | `framework/server/routes/graph.py:93-112` |
| `usage` in chat completions is session-cumulative, not per-request | `routes/openai.py:419-428` |
| Manifest `vertices`/`edges` entries may be absolute or `../` paths — arbitrary JSON read outside the manifest tree | `framework/graphs/loader.py:48-59` |
| `has_cycle` reports dangling edges as cycles, and `compute_dag_tiers` then silently falls back to insertion order | `framework/graphs/validation.py:74-78,127-132` |
| Missing `Union`/`Path` imports in `executor_v4.py` — `typing.get_type_hints()` fails (verified `NameError`) | `framework/executor_v4.py:15,93,170` |
| `stream()`/`run()` re-run on the same executor returns stale/truncated results with zero edge executions | `framework/executor_v4.py:154,566-587,844` |
| `step()` never drains the event queue (unbounded growth) and bypasses group/per-edge concurrency limits | `framework/executor_v4.py:589-646` |
| Callable-script edges break when cloned/snapshotted: `to_dict()` emits the bare function name | `framework/edges/base.py:176-180`; `framework/graphs/subgraph_ops.py:154,247,333` |

---

## 5. Architecture & design assessment

**What is good**
- The vertex-state / edge-compute separation, the two-sided handshake, and the staging scratchpad are a
  coherent model, and the code documents its own invariants well.
- SQL hygiene is clean: all values are parameterized (the only dynamic SQL, `vertex_v4.py:772/793/877`,
  joins static clause strings and coerces the limit with `int()`). No injection found.
- WAL/commit/rollback handling in `VertexStoreV4` was checked and is correct for the single-loop case.

**What needs attention**
1. **Two full frameworks coexist.** V4 core ≈ 9,661 lines; the strictly-V1 stack is another 4,660 lines
   (`framework/vertex.py`, `edge.py`, `graph.py`, `subgraph.py`, `pipeline.py`, `serve/`, `executor/`,
   `builders/`), plus ~2,200 lines of shared `agents/`+`utils/` code that both generations import.
   31 example files import V1 and 14 test modules target V1, so it is still a live, maintained surface.
   It is also the direct cause of H15 (V1 configs advertised as V4 models).
   **Recommendation**: declare V4 the only supported engine, move V1 to `legacy/` or delete it, and migrate
   the examples that still matter.
2. **Layering inversion**: `framework/server/routes/openai.py:270,273` and `routes/tools.py:73,84,88`
   import from `examples.dynamic_tool_library.tool_scripts`. The package does not ship `examples/`
   (`pyproject.toml:29-30`), so these imports fail after installation, and the unguarded fallback at
   `tools.py:88` raises `ImportError` (500).
3. **`framework/__init__.py:76` eagerly imports the FastAPI server**, so importing the engine requires
   FastAPI even for pure batch use — and `pyproject.toml:15-20` does not declare it (§6).
4. **Enum overloading**: `VertexStateV4` mixes lifecycle states with DFS colors, aliases the enum as
   `NodeColor`, and overrides `__eq__` to compare against ints (`framework/vertex_v4.py:24-56`). This is
   the root cause of the state-corruption defect and it is baked into the SQLite `CHECK` constraint.
5. **Snapshot granularity**: snapshotting the full graph on every mutation/edge is O(N²) I/O and has no
   retention. Consider WAL-style deltas or an explicit `--snapshot-every` policy.
6. **Function size**: `serve/app.py:239 create_app()` (359 lines), `routes/openai.py:197`
   (324 lines), `http_executor_v4.py:214 step()` (323 lines), `executor_v4.py:320`
   `_execute_single_edge()` (179 lines), `edges/base.py:209 from_config()` (167 lines).
7. **No LICENSE file** although `pyproject.toml:9` declares MIT.

---

## 6. Packaging & documentation accuracy

| Claim / issue | Evidence |
| :--- | :--- |
| `pip install -e .` (README §1 step 2) yields an unimportable package | `pyproject.toml:15-20` declares only httpx/socksio/tenacity/pydantic, but `framework/__init__.py:76` imports FastAPI. Verified: with only the declared deps, `import framework` → `ModuleNotFoundError: No module named 'fastapi'`. `fastapi`, `uvicorn`, `beautifulsoup4` are used by the package but undeclared |
| `aiohttp>=3.9` declared but unused | zero imports anywhere in the repo (`requirements.txt:10`) |
| Python version mismatch | `README.md:11` says 3.12+; `pyproject.toml:13` says `>=3.10` |
| `.env` is never loaded | no `dotenv`/manual parser anywhere; `SENSENOVA_API_KEY` etc. are read only from the real environment (`sensenova_edge_v4.py:139`). `.env.example` tells users to create `.env` and fill it in — those values are ignored |
| Stale absolute paths to another machine | `agent.md:141,165` and `docs/review_report.md` (many lines) link `file:///home/gekkasayu/vertex_edge_agent/...` |
| Duplicated review doc | `docs/review_report.md` and `examples/review_report.md` are byte-identical (md5 `0d5180cb…`) and both describe a pre-refactor layout |
| `examples/README.md` inaccuracies | documents `python examples/sensenova_v4/run.py` (file does not exist; only `demo.py`), claims "19 executable examples" (there are 24 dirs), and claims every example has a README (`sensenova_v4/` has none) |
| `docs/ARCHITECTURE.md:41` documents states `running`/`error` and attributes `gate`/`fan-in`/`streaming` | none of these exist in `VertexStateV4`/`VertexAttributeV4` |
| `docs/HANDOFF.md:11` claims "551 tests" | current collection is 564 (558 non-live + 6 live) |
| Test artifacts pollute the repo | `snapshots/` is gitignored but the suite writes ~565 files / 3.4 MB there |
| 61 tracked Markdown files vs 164 Python files | heavy, drifted documentation surface |

**Caution**: the working tree contains a real-looking `HF_TOKEN` in `.env`. It is gitignored and was never
committed (the `.env` files in history belong to an earlier project generation and contain only
`ROOT_PATH`/`UTILS_PATH`), but it is present in plaintext on disk and should be rotated if the machine is
shared.

---

## 7. Test suite & CI assessment

**Result**: `pytest tests/ -m "not live"` → **558 passed, 6 deselected, 0 failed** in 85.8 s. The
`docs/HANDOFF.md` "100% pass rate" claim is accurate. That is precisely the problem: the suite passes
while three critical engine bugs and the state-corruption defect are present.

- **The suite enshrines a defect.** `tests/test_v4_layer3_graph.py:82-84` asserts
  `v1.state == VertexStateV4.BLACK.value` — i.e. that validation overwrites the lifecycle state. Any fix
  must update this test first.
- **Coverage is inverted relative to risk.** `test_agents.py` (73 tests) and
  `test_opencode_and_proxied_agents.py` (57) dominate; the core scheduler `executor_v4.py` has 6 tests
  (`test_v4_layer4_executor.py`), the harness executor has 2 (`test_harness_http_executor.py`), and the
  server layer has 5 (`test_v4_layer5_server.py`). The fan-in barrier, concurrency groups, retry/circuit
  breaker and cancellation paths are effectively untested — which is why E1/E2/H8/H9 all pass CI.
- **No mixed-strategy fan-in test.** Every fan-in test uses all-participating edges, the one configuration
  that works (E1).
- **No security tests.** Nothing asserts that mutation endpoints require credentials, that `script` is
  rejected, or that manifest paths are confined.
- **CI scope** (`.github/workflows/ci.yml`): runs only on pushes/PRs to `vertex-edge-agent`, installs
  `requirements.txt`, runs `pytest tests/ -v -m "not live"`. No linting, no type checking, no coverage
  gate, and no job that would catch the packaging break in §6 (it never runs `pip install -e .`).
- **Artifact pollution**: the suite writes into the repository `snapshots/` directory rather than a
  `tmp_path` fixture — one full run created 354 files there (565 total, 3.4 MB).
- **Live tests hang without network**: `pytest -m live` selects `test_sensenova_live.py` and
  `test_v4_layer6_e2e.py`, which block on real HTTP calls with no short timeout. CI correctly deselects
  them (`-m "not live"`), but a developer running `pytest tests/` locally will appear to hang.

---

## 8. Prioritized remediation plan

**P0 — before any non-localhost deployment**
1. Add an API-key dependency to every non-dashboard router; default CORS to `[]`; refuse non-loopback binds
   without a token. (C3)
2. Remove `eval` of inline lambdas; confine `script` to an allow-listed directory with no absolute paths. (C1)
3. Route every client-supplied manifest path through `validate_path_security`, or remove
   `vea_manifest_path`/`subgraph_manifest` from the request schemas. (C2, C5)
4. Remove `base_url`/`api_key`/`model` from `RouteAndRunRequest`, or allow-list them. (C4)
5. Validate `session_id` and assert the resolved snapshot path stays inside the base dir. (C5)

**P1 — engine correctness (data loss / non-termination)**
6. Fix the fan-in accumulation to use participating edge ids (E1), stop recording failures in
   `_fan_in_completed` (E2), and cap the harness loop (E3).
7. Refuse/skip tool edges in `ExecutorV4` (H1); propagate subgraph failure (H2).
8. Fix `LLMToolEdgeV4` construction/serialization and persist `tools` (H3); fix the
   `add_or_update_edge` settings flattening (H4) and the 3-type whitelist (H5).
9. Make `except TypeError` not re-run the edge body (H9); check `sub_res`/cancelled tasks before reporting
   success (H8); surface cancelled edges in `errors`.

**P2 — state & data integrity**
10. Move DFS colors out of `VertexStateV4`/`vertex.state` into `graph.node_states` only; drop the DB
    `CHECK` values (needs a migration) and update `test_v4_layer3_graph.py`.
11. Make merges fail loudly on malformed input / unknown strategy; make upserts patch-style; wrap
    fan-in merges in `BEGIN IMMEDIATE`.
12. Make subgraph ops atomic and collision-safe.

**P3 — platform health**
13. Consolidate on V4 and retire/relocate the V1 stack; remove `framework → examples` imports.
14. Declare all runtime deps in `pyproject.toml` and make `import framework` not require FastAPI.
15. Snapshot retention/compaction; staging/metrics pruning; session/lock eviction.
16. Offload SQLite via `asyncio.to_thread`/aiosqlite, or document the single-loop constraint explicitly.
17. Fix the OpenAI contract: 404 unknown model, correct streaming deltas, per-request usage, working
    `per_tier`, multimodal handling.
18. Refresh the docs listed in §6 and add a LICENSE file.

**P4 — test strategy**
19. Add regression tests for E1–E3, H1–H4 and the five P0 security issues (they should fail today).
20. Add a `pip install -e . && python -c "import framework"` CI job, plus lint/type-check.
21. Move test artifacts to `tmp_path`; add a coverage gate focused on `executor_v4.py`, `http_executor_v4.py`
    and `vertex_v4.py`.
