# Forum AI Report (MapEdge)

> Documented following the "Problem / Solution / Changes / Verification" format: MapEdge forum AI digest experiment compared against direct agent execution.

## Issue 1: 8-Way Manual Fan-out vs Single MapEdge

### Problem
The initial report generator manually configured 8 fan-out paths for fetching and summarizing threads (`e_sel1-8` / `e_fetch1-8` / `e_sum1-8`), resulting in large configurations, fixed concurrency, and poor reusability.

### Solution
Apply the MapEdge pattern: a single `ProcessThreadsMap(MapEdge)` with `settings.pipeline` (fetch -> summarize) concurrently processing each filtered thread, throttled by `max_concurrency`.

### Changes
- `examples/s1_ai_report_map/config.json`: Declares `script: s1_edges.py:ProcessThreadsMap` with a two-step pipeline.
- `examples/s1_ai_report_map/s1_edges.py`: Implements `FetchEdge`, `SummarizeEdge`, and `ProcessThreadsMap`.

### Verification
- **Test Plan**: Verify MapEdge executes the pipeline across filtered threads and aggregates outputs.
- **Method**:
  `python examples/s1_ai_report_map/demo.py` (proxy declared in config).
- **Result**: `report.md` generated successfully across concurrent thread pipelines.

## Issue 2: 24-Hour Time Window Excludes Original Thread Starters

### Problem
MapEdge operates over a recent 24-hour reply window; older original starter posts (e.g. historical setup tutorials) fall outside this window. Direct single-agent page fetching captures the full page archive.

### Solution
Document as an intentional design trade-off: the 24-hour window captures active discussions rather than historical archives.

### Changes
- Documented in `ai_report_notes.md` and comparison tables.

### Verification
- **Test Plan**: Compare identical threads across MapEdge and direct runner (`opencode run --model opencode/hy3-free`).
- **Method**: Run both pipelines on the same threads and compare content coverage and token costs.
- **Result**: MapEdge reports retain specific reply numbers and user attributions; direct execution retains full historical archives; MapEdge cost is ~27% of direct execution (see table in `ai_report_notes.md` Issue 6).

## Files

- `config.json`, `demo.py`, `s1_edges.py`, `vertex/report_hook.py`, `report.md`
- `opencode_direct.md`: Comparative baseline report generated via direct single-agent execution.
