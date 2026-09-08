# HN AI Report V4 — Legacy MapEdge → V4 Migration

> **Problem / Solution / Changes / Verification** format

## Problem: MapEdge Fan-Out Doesn't Exist in V4

The legacy [HN AI Report](../hn_ai_report/) uses `MapEdge` — a fan-out edge that dynamically spawns per-story sub-edges (`FetchCommentsEdge` → `SummarizeEdge`). V4 removed `MapEdge` in favor of explicit topological fan-out via multiple edges from a single vertex.

## Solution: Single CodeEdgeV4 with Internal Parallelism

Replace `ProcessStoriesMap(MapEdge)` with one `CodeEdgeV4` (`summarize_all`) that:

1. Parses the AI-filtered story list from upstream
2. Fetches comments for all stories in parallel (no semaphore — HN API is fast)
3. LLM-summarizes each story in parallel (with semaphore to respect rate limits)
4. Assembles the final markdown report

The V4 pipeline is a simple linear DAG:

```
v_start ──[e_fetch]──▶ v_stories ──[e_filter (LLM)]──▶ v_ai_stories ──[e_summarize]──▶ v_report
 (data ready)          (top stories)     (JSON, AI only)         (markdown report)
```

## Architecture Comparison

| Aspect | Legacy (v1) | V4 |
|---|---|---|
| **Fetch** | `FetchTopStoriesEdge(Edge)` | `CodeEdgeV4` with async script |
| **Filter** | `FilterEdge(Edge)` + LLM | `LLMEdgeV4` with prompt template |
| **Fan-out** | `ProcessStoriesMap(MapEdge)` | Single `CodeEdgeV4` — internal `asyncio.gather` |
| **Summarize** | `SummarizeEdge(Edge)` per story | Loop inside `summarize_all()` |
| **State** | In-memory dicts | SQLite `VertexStoreV4` with WAL |
| **Persistence** | None | Full — crash recovery, reentry, replay |
| **Handshake** | Implicit (data flows forward) | Explicit: upstream `data_ready` → downstream `todo` |
| **Edges** | 3 edges + MapEdge pipeline | 3 edges (code → llm → code) |

## Migration Mapping

```
Legacy:                        V4:
─────────────────────────────  ─────────────────────────────────
Edge (base class)              CodeEdgeV4 / LLMEdgeV4
MapEdge fan-out                CodeEdgeV4 + asyncio.gather
EdgeSignal (COMPLETED/FAILED)  VertexStateV4 (data_ready/reject)
VertexState (IDLE/RUNNING)     VertexStateV4 (idle/todo/data_ready/reject/forbidden)
MapEdge pipeline steps         Single function with internal loop
script: "file.py:ClassName"    script: "file.py:function_name"
```

## Files

| File | Purpose |
|---|---|
| `config.json` | V4 graph manifest (vertices + edges) |
| `hn_edges_v4.py` | Edge functions (fetch, summarize) |
| `demo.py` | Standalone execution entry point |

## Run

```bash
# Basic execution
python examples/hn_ai_report_v4/demo.py

# With more concurrency and proxy
python examples/hn_ai_report_v4/demo.py --concurrency 3 --limit 15

# With API key (for authenticated LLM endpoints)
export SENSENOVA_API_KEY=sk-your-key
python examples/hn_ai_report_v4/demo.py

# With HN API proxy
export HN_PROXY=http://127.0.0.1:7890
python examples/hn_ai_report_v4/demo.py
```

## Verification

```bash
# Run and check output
python examples/hn_ai_report_v4/demo.py 2>&1 | tail -30

# Verify report structure
head -20 examples/hn_ai_report_v4/report.md

# Check story count
grep -c "^## " examples/hn_ai_report_v4/report.md
```

## Key Differences from Legacy

1. **No MapEdge** — V4 has no dynamic fan-out. The `summarize_all()` function handles parallelism internally with `asyncio.Semaphore`.
2. **SQLite persistence** — All vertex states and content are persisted to SQLite. Crash recovery is built-in.
3. **Explicit handshake** — Each edge checks `check_handshake()` before executing. Upstream must be `data_ready`, downstream must be `todo`.
4. **JSON attribute** — The `v_ai_stories` vertex has `json` attribute, so `LLMEdgeV4` automatically validates JSON syntax before delivering.
5. **Staging** — Intermediate artifacts (prompt drafts, error feedback) are staged to `session_staging` table for observability.
