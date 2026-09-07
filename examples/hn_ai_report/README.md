# HN AI Report — End-to-End AI Digest (MapEdge)

> Documented following the "Problem / Solution / Changes / Verification" format: Hacker News AI digest experiment demonstrating the MapEdge architecture (fetch -> filter -> concurrent fetch and summarize per story -> aggregated `report.md`).

## Issue 1: 12 Vertices and 17 Edges in Manual Fan-out was Too Heavy

### Problem
Earlier HN implementations manually declared every edge path (independent fetch/summarize edges per story), producing 120+ lines of config with rigid concurrency. Furthermore, `hn_edges.py` and inline config code previously coexisted, causing architectural ambiguity.

### Solution
Adopt the "Class Extension + MapEdge" architecture:
- Encapsulate all custom logic in subclasses within `hn_edges.py` (`FetchTopStoriesEdge` / `FilterEdge` / `FetchCommentsEdge` / `SummarizeEdge` / `ProcessStoriesMap`).
- Reference subclasses explicitly in config using `script: hn_edges.py:ClassName` and remove all inline code snippets.
- Consolidate per-story processing into a single `ProcessStoriesMap(MapEdge)` step driven by `settings.pipeline`.

### Changes
- `examples/hn_ai_report/config.json`: Compact configuration (~40 lines); explicit `script` class references; proxy configured in settings.
- `examples/hn_ai_report/hn_edges.py`: Dedicated edge subclasses.
- `examples/hn_ai_report/demo.py`: Standalone execution entry point.

### Verification
- **Test Plan**: Generate end-to-end HN AI digest report.
- **Method**:
  `env -u HTTPS_PROXY -u HTTP_PROXY python examples/hn_ai_report/demo.py` (proxy defined in config).
- **Result**: `report.md` generated (~99 lines); stories filtered dynamically; markdown formatted with `# [Title](Link)`.

## Issue 2: Script Relative Paths and Explicit Class Names in MapEdge Pipeline

### Problem
Pipeline step `script` paths previously resolved against CWD (causing "Script not found" errors when executed from different directories); automatic discovery previously matched subclasses alphabetically, picking the wrong class (e.g. `SummarizeEdge` shadowed by `FetchEdge`).

### Solution
- Normalize step `script` paths relative to the configuration file directory (`framework/graph.py`).
- Prioritize explicit class name resolution in `load_class_from_script` (`framework/utils/script_loader.py`).

### Changes
- `framework/graph.py`, `framework/utils/script_loader.py`: Path normalization and explicit class resolution.

### Verification
- **Test Plan**: Load pipeline steps with correct class resolution regardless of current working directory.
- **Method**:
  `pytest tests/test_script_loader.py -q` + standalone demo execution.
- **Result**: Passed; report includes per-story summaries (`SummarizeEdge.post_process` executed).

## Issue 3: Token and Latency Comparison (MapEdge vs Direct Agent Run)

### Problem
Needed to quantify token usage and cost differences between MapEdge (framework pipeline) and direct CLI execution (`opencode run`).

### Solution
Execute both pipelines against the same HN stories and collect usage metrics (via framework `get_usage_summary()` and agent telemetry logs).

### Changes
- `framework/agents/_http_base.py`: Token usage tracking (`usage_log` / `get_usage_summary`). Full comparative benchmark detailed in `ai_report_notes.md` Issue 6.

### Verification
- **Test Plan**: Benchmark token usage and latency between both approaches.
- **Method**: Run demo script and direct runner on the same input dataset; compare telemetry.
- **Result**: MapEdge total token consumption was ~**32%** of direct execution (HN), with input tokens 6-9k vs 78k (direct fetch ingested full raw page); see table in `ai_report_notes.md`.

## Files

- `config.json`, `demo.py`, `hn_edges.py`, `vertex/report_hook.py`, `report.md`, `opencode_direct.md`
