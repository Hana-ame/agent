# VEA v4 Redesign RFC — Extension Surface and Remaining Structural Work

- **Status**: implemented for the four review P0/P1 items; this document records
  what changed, what deliberately did not, and what is left.
- **Context**: `docs/CODE_REVIEW_2026-09-08.md`

---

## 1. The extension contract (the part that was broken)

**Intended workflow** (and now the actual one):

```python
# my_edges.py
class MyEdge(EdgeV4):
    def __init__(self, edge_id, input_vertex, output_vertex, settings=None): ...
    async def run(self, session_id, store, agent=None, auto_transition=True, **kwargs): ...
```

```json
{ "id": "e1", "type": "my_edges.py:MyEdge", "input_vertex": "a", "output_vertex": "b" }
```

No framework edit, no registration call. Runnable reference: `examples/custom_edge/`.

### What blocked it before

`EdgeV4.from_config` was a closed set of hardcoded `elif` branches, and the same
type→class mapping was duplicated in **five** places (`edges/base.py`,
`graphs/loader.py` twice, `graph_manager_v4.py`, `edges/cli.py`,
`server/routes/openai.py`). Adding a type meant editing all of them; `LLMToolEdgeV4`
was added and broke in two. There was no path for user classes at all, even though
the legacy V1 layer had one (`load_class_from_script`).

### What it is now

Two tiers, one resolution point:

1. **Registry** (`framework/edges/registry.py`) — built-ins register once with
   `@register_edge_type("code", "code_edge")`. Every dispatch site reads the registry.
2. **Dynamic path** — any other `type` containing `:` or ending in `.py` is loaded
   from the script and instantiated, with constructor kwargs filtered by signature.

| Behaviour | Detail |
| :--- | :--- |
| Resolution order | explicit class → registry → script spec |
| Relative specs | resolved against the manifest/edge JSON directory, then the configured script roots (`--script-root` / `VEA_SCRIPT_ROOTS`) |
| Constructor mapping | `_filter_kwargs` drops params the custom `__init__` does not accept |
| Custom mapping | override `from_config_dict(data, base_dir)` |
| Round-trip | `to_dict()` keeps your `type` and records `settings._edge_class_spec` (the path the class was actually loaded from) so SQLite/snapshot restore works from any cwd |
| Module loading | cached by `(path, mtime)`; `sys.path` is not permanently mutated |
| API boundary | script specs validated against allowed roots; inline `lambda` disabled by default |
| Client surface | `GET /api/edge-types` lists the registry; the dashboard type field is free text with suggestions, so `my_edge.py:MyEdge` is usable without a code change |
| Capability flags | `EMITS_TOOL_CALL`, `IS_RECOVERY` replace scattered `isinstance` checks |

---

## 2. Implemented structural changes

| # | Change | Where | Review refs |
| :-- | :--- | :--- | :--- |
| A | API-key auth on `/api/*` + `/v1/*`, CORS off by default, refuse non-loopback bind without a key | `server/security.py`, `server/app.py` | C3 |
| A | Inline `eval` of scripts removed (opt-in `settings.allow_inline_script`), script roots confined, API rejects inline/out-of-root specs | `edges/code.py`, `edges/reflexive.py`, `utils/script_loader.py` | C1 |
| A | `vea_manifest_path` removed; `subgraph_manifest` and nested manifests confined to a base dir | `routes/openai.py`, `routes/graph.py`, `workflow_executor_v4.py` | C2, C5 |
| A | `session_id` validated; snapshot paths asserted inside the base dir | `utils/paths.py`, `snapshot_v4.py`, `server/security.py` | C5 |
| A | Client `base_url`/`api_key`/`model` removed from `RouteAndRunRequest` (SSRF) | `server/schemas.py`, `routes/tools.py` | C4 |
| B | Edge-type registry + capability flags; `LLMToolEdgeV4` constructs and serializes; `tools` persisted; settings reach the live edge | `edges/registry.py`, `edges/*`, `graphs/loader.py`, `graph_manager_v4.py` | H3–H5 |
| B | `framework` no longer imports `examples/` (router moved to `framework/tool_catalog/`) | `framework/tool_catalog/` | §5 |
| C | `VertexStateV4` split from `TraversalColor`; validation no longer writes colors into vertex state; SQLite migration v1 maps legacy colors to `idle` | `vertex_v4.py`, `graphs/validation.py`, `graphs/core.py` | §4 |
| D | Fan-in settlement rewritten: forward-edge barrier, no permanent blacklist, no early-settle cancellation, no body re-execution, no fabricated tool results, subgraph failures propagate | `executor_v4.py`, `workflow_executor_v4.py` | E1–E3, H1, H2, H8, H9 |
| E | Zero-registration custom edges (this document) | `edges/base.py`, `graphs/loader.py` | extensibility |
| F | Extension surface completed end-to-end: `GET /api/edge-types`, dashboard type field is free text (was a 4-option `<select>` that silently reset `llm_tool`/custom edges to `code`), `script_roots` actually reaches validation and the loader, relative specs resolve against configured roots, `_edge_class_spec` records the real module path, load failures return 400 not 500 | `server/routes/dashboard.py`, `templates/dashboard.*`, `graph_manager_v4.py`, `edges/base.py`, `utils/script_loader.py` | H3 follow-up |
| F | Dashboard vertex state list completed (`pruning` was missing) and `setSelectValue` appends unlisted values instead of silently emptying the select | `templates/dashboard.html`, `templates/dashboard.js` | state/color follow-up |
| F | Packaging fixed: `fastapi`/`uvicorn` declared (they were undeclared while `import framework` requires them), dashboard assets shipped via `package-data`, `vea-server` console script; wheel install verified standalone | `pyproject.toml`, `framework/templates/__init__.py` | §4.7 |
| F | V4 entry point added without touching the V1 runner: `framework/run_v4.py` (`run_from_manifest` / `run_manifest_async` / `load_v4_manifest`) collapses `load_from_manifest` + `populate_store` + `ExecutorV4.run()` into one call; `vea-run-v4` console script and the thin `examples/run_v4.py` wrapper; both runners reject the other stack's manifests with a message naming the correct one; `--script-root` reachable from the CLI | `framework/run_v4.py`, `examples/run_v4.py`, `pyproject.toml`, `tests/test_run_v4.py` | usability follow-up |
| F | Custom edges are `script.py:ClassName` everywhere — no type whitelist. `framework/edges/cli.py --type` dropped `choices=edge_type_choices()` and passes the value through to `EdgeV4.from_config`, so `--type my_edge.py:MyEdge` works; a typo now fails at the edge layer with a message explaining the script-spec format instead of an argparse "invalid choice". `parse_args(argv=None)` added for testability | `framework/edges/cli.py`, `tests/test_edges_cli_type.py` | usability follow-up |
| F | Standalone edge CLI flag names match the config JSON keys: `--id` replaced `--edge-id` (kept as a deprecated alias — `tests/test_v4_system.py` still uses the old spelling), `--input` / `--output` keep their short names but the help text now names the JSON keys they map onto (`input_vertex` / `output_vertex`), and the canonical config key for the session is `"session"` (`"session_id"` kept as a deprecated alias). `--db` / `--seed-input` / `--mock` / `--dir` are runner parameters — consumed by the CLI itself, not by `EdgeV4.from_config` — which is why they only appear in a single-edge `--config`, never in a graph manifest | `framework/edges/cli.py`, `tests/test_edges_cli_type.py` | usability follow-up |
| F | `python -m framework.edges.cli` no longer prints runpy's "already in sys.modules" RuntimeWarning: `framework/edge_v4.py` imported `framework.edges.cli` at module level, and `framework/__init__.py` imports `edge_v4`, so the CLI module was already in `sys.modules` before runpy executed it. The `main` / `parse_args` re-export now resolves through a PEP 562 `__getattr__` (backwards compatible — `from framework.edge_v4 import main, parse_args` still works), matching what the four LLM edge modules already did inside their `__main__` blocks | `framework/edge_v4.py`, `tests/test_edges_cli_type.py` | extensibility follow-up |
| F | Core engine no longer needs the web stack: `create_v4_server`, `SessionGraphManagerV4`, `SSEExecutorV4` and `ToolCallEcho` resolve lazily through a single PEP 562 `__getattr__` map (they are the only two modules — `server_v4`, `sse_executor_v4` — that import FastAPI). `import framework` now works with FastAPI absent; symbols resolve to the same objects and still work with it installed | `framework/__init__.py`, `tests/test_edges_cli_type.py` | §4.7 |
| F | CI now tests the *installed* distribution, not just the source tree. The existing job only ran pytest against the checkout, so it could never see an undeclared dependency, a broken console script or assets missing from the wheel (all three happened). A new `install-and-smoke` job does an editable install, runs the console scripts from outside the checkout, asserts wheel contents and entry points, installs the wheel into a clean venv and serves the dashboard from the packaged assets, then uninstalls FastAPI and imports the core engine | `.github/workflows/ci.yml` | §4.7 |
| F | The session key is now `"session"` everywhere a document declares one: `graphs/loader.py` reads `"session"` (canonical) with `"session_id"` kept as a deprecated alias, and the three example manifests (`examples/hn_v4`, `subgraph_v4` parent/child) now use the canonical key. The internal storage fields (`VertexStoreV4`, `EdgeRecordV4`) and the HTTP request bodies still use `session_id` — a different layer, deliberately untouched | `framework/graphs/loader.py`, `examples/hn_v4/manifest.json`, `examples/subgraph_v4/`, `tests/test_run_v4.py` | naming follow-up |

Verification: `pytest tests/ -m "not live"` → **713 passed, 0 failed** (6 live
deselected). New regression suites: `tests/test_security_hardening.py` (32),
`tests/test_edge_registry.py` (29), `tests/test_state_color_split.py` (9),
`tests/test_executor_settlement.py` (8), `tests/test_custom_edge_extension.py` (13),
`tests/test_edge_type_surface.py` (16), `tests/test_run_v4.py` (31),
`tests/test_edges_cli_type.py` (18). The built wheel was installed into a clean
target and served the dashboard with a non-empty body, and the core engine was
re-imported with FastAPI uninstalled — both checks now run on every push in the
`install-and-smoke` CI job.

---

## 3. Deliberately not changed

- **V1 stack left in place** (`vertex.py`, `edge.py`, `graph.py`, `subgraph.py`,
  `serve/`, `executor/`, `builders/`; ~4.7k lines, 14 test modules, 31 example
  imports). Retiring it is a product decision, not a bug fix, and it would break
  the examples that still target it.
- **`serve/app.py` (legacy V1 server)**: same hardening was *not* applied. It is a
  separate application; if it is still deployed it needs its own pass.
- **`per_tier` semantics, SSE chunk shape, `finally: yield`** (H14/H15-adjacent):
  contract fixes with client-visible behavior; scoped out of this round.

---

## 4. Remaining structural work (ranked)

1. **Async store** — every SQLite call runs on the event loop
   (`vertex_v4.py` sync API, no `to_thread`). Under concurrency the loop stalls for
   up to `busy_timeout` (5 s). Either move to `asyncio.to_thread`/aiosqlite or
   document the single-loop constraint and pin one worker.
2. **Snapshot policy** — a full graph is dumped on every mutation and every edge
   completion, with no retention. Add `--snapshot-every N` / delta snapshots and a
   prune API; the repository currently accumulates ~565 files per test run.
3. **V4 hooks** — V1 had `ExecutorHooks`; V4 has only the async event stream. Add
   `on_edge_start/complete/fail` callbacks so tracing and metrics enrichment do not
   require editing `executor_v4.py`.
4. **OpenAI contract** — 404 unknown model, correct streaming deltas, per-request
   `usage`, working `per_tier`, multimodal message handling.
5. **SSE robustness** — keepalive frames, `X-Accel-Buffering` everywhere, and fix
   `finally: yield` (`server/sse.py`).
6. **V1 retirement** — after migrating the examples worth keeping.
7. **Packaging** — ~~declare `fastapi`/`uvicorn`/`beautifulsoup4`~~ (done: deps,
   dashboard `package-data`, `vea-server` / `vea-run-v4` console scripts, and a
   wheel install verified). ~~make `import framework` not require FastAPI~~ (done:
   the four server/SSE symbols resolve lazily, so the core engine imports without
   the web stack — a lean wheel is now possible). ~~add a `pip install -e .` CI
   job~~ (done: `.github/workflows/ci.yml` gained an `install-and-smoke` job —
   editable install, console scripts run from outside the checkout, wheel
   contents and entry-point assertions, a clean-venv install that serves the
   dashboard from the packaged assets, then a core import with FastAPI
   uninstalled). Packaging is closed out.
8. **Retention/eviction** — session locks/graphs grow unbounded; staging and
   `edge_metrics` have no prune path.

---

## 5. Migration notes for users

| Before | After |
| :--- | :--- |
| `python3 -m framework.server_v4 --host 0.0.0.0` | requires `--api-key` or `VEA_API_KEY` |
| Any browser could call the API | `/api/*` and `/v1/*` require `X-API-Key` when configured; dashboard prompts once |
| `{"vea_manifest_path": "/abs/path.json"}` | removed; use a registered model template or a confined `subgraph_manifest` |
| `"script": "lambda x: x"` in JSON | rejected unless the config sets `settings.allow_inline_script = true` (never allowed over HTTP) |
| `NodeColor is VertexStateV4` | `NodeColor is TraversalColor`; `VertexStateV4` has no color members |
| Old DB with `white`/`gray`/`black` states | auto-migrated to `idle` on first open (`PRAGMA user_version = 1`) |
| Adding an edge type = 5 file edits | subclass + `"type": "file.py:ClassName"`, or one `@register_edge_type(...)` |
| Dashboard type dropdown (4 options) | free text with suggestions from `GET /api/edge-types`; script specs accepted |
| `create_v4_server(script_roots=...)` was ignored | wired through to validation, loading and graph rehydration; CLI flag `--script-root` |
