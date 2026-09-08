# AI Report Examples (MapEdge Architecture) Changelog and Notes

> Documents architectural issues, resolutions, and benchmark findings across the report pipeline examples (`s1_ai_report_map`, `hn_ai_report`, `finance_ai_report`). All token metrics and latency figures reflect empirical benchmark runs.

---

## Issue 1: Pipeline step `script` relative path resolution

### Problem
MapEdge's `settings.pipeline[].script` (e.g. `hn_edges.py:FetchCommentsEdge`) resolved relative to current working directory (CWD), failing with `Script not found` when executed from the repository root.

### Solution
Normalize pipeline step `script` paths relative to the directory containing the configuration file (`base_dir`).

### Changes
- `framework/graph.py` (`from_dict`): Normalize `settings.pipeline` steps with `os.path.join(base_dir, step_script)` for relative paths.

### Verification
- Executed `examples/hn_ai_report/demo.py` from root and parent directories; successfully generated `report.md` without path errors.

---

## Issue 2: Script loader auto-discovery alphabetical selection error

### Problem
`load_class_from_script("s1_edges.py:SummarizeEdge", ...)` relied on alphabetical class discovery. `FetchEdge` ordered before `SummarizeEdge`, causing the pipeline to load the wrong class and skipping `post_process`.

### Solution
Lookup requested class name explicitly with `getattr(module, cls_name)` before falling back to auto-discovery.

### Changes
- `framework/utils/script_loader.py`: Prioritize explicit class resolution when class name is provided.

### Verification
- Regression test `tests/test_script_loader.py` validates that `SummarizeEdge` loads correctly rather than `FetchEdge`.

---

## Issue 3: Forum crawler timestamp and empty rating element parsing

### Problem
- Post selector `div[id^="post_"]` matched empty rating placeholders `id="post_rate_div_<pid>"`.
- Timestamp strings lacked `span[title]` attributes, causing failed datetime parsing, dropping posts under the 24-hour cutoff and producing empty reports.

### Solution
- Narrow container selector to `^post_\d+$`.
- Strip locale prefix and apply regex extraction for `YYYY-M-D H:M`.
- Sort replies chronologically.

### Changes
- `examples/s1_ai_report/s1_edges.py`, `examples/s1_ai_report_map/s1_edges.py`, `tests/fixtures/s1_thread.html`.

### Verification
- Tested against offline fixture in `tests/test_s1_edges.py`. All posts and timestamps parsed accurately.

---

## Issue 4: Title restatement token overhead

### Problem
Having the LLM restate thread titles in summaries consumed output tokens unnecessarily and introduced potential hallucination.

### Solution
Structured titles: attach `title` and `url` from crawler output to edge results; LLM generates only summary content. `ReportVertex.on_receive` formats markdown with structured fields.

### Changes
- `SummarizeEdge.post_process` returns structured dict `{"title", "url", "summary"}`.

### Verification
- Output reports verify headings formatted directly from source data.

---

## Issue 5: Dynamic candidate selection without artificial limits

### Problem
Arbitrary caps (e.g. "top 3 only" or "max 120 words") artificially truncated valid information.

### Solution
Prompt instructs model to select all relevant candidate threads matching topic criteria without hardcoded caps.

### Changes
- Updated prompt configurations in `examples/hn_ai_report/config.json`, `examples/s1_ai_report_map/config.json`, and `examples/finance_ai_report/config.json`.

### Verification
- Empirical test runs select varying thread counts (3-8 threads) depending dynamically on daily volume.

---

## Issue 6: Token and Latency Profiling (MapEdge vs Single Agent)

### Problem
Needed quantitative measurement of cost and performance trade-offs between structured MapEdge fan-out and monolithic single-agent processing.

### Solution
Benchmarked identical thread sets across both architectures, recording token usage and elapsed time.

### Benchmark Results

| Metric | Forum MapEdge | Forum Single-Agent | HN MapEdge | HN Single-Agent |
|---|---|---|---|---|
| Latency | 321.5s (3 calls) | 418.2s (1 agent) | 101.6s (6 calls) | 304.0s (1 agent) |
| Total tokens | 19,804 | 73,574 | 25,949 | 80,235 |
| Input tokens | 6,108 | 67,882 | 9,163 | 78,035 |
| Completion tokens | 13,696 | 5,692 | 16,786 | 2,200 |
| Reasoning tokens | 12,633 | 6,354 | 15,034 | 2,545 |
| Visible tokens | 1,063 | 5,692 | 1,752 | 2,200 |

**Finding**: MapEdge pipeline consumed **27% to 32%** of single-agent input tokens because pre-processing extracts only relevant comment threads rather than submitting entire web pages.

---

## Issue 7: Proxy and Endpoint Specifications

### Problem
Implicit environment variables (`HTTPS_PROXY`) caused configuration drift across environments.

### Solution
- Explicitly declare proxies in `settings.https_proxy`.
- Require complete base URLs including path (`/chat/completions`) without automatic suffix guessing.

### Changes
- `framework/agents/_http_base.py`: Direct endpoint resolution and proxy connection caching.

### Verification
- Verified execution with and without global proxy environment variables.

---

## Issue 8: Fetch Stage Timeout Controls

### Problem
External network requests without explicit timeouts risked hanging entire workflows indefinitely.

### Solution
Added `settings.timeout` (default 30s) per pipeline step.

### Changes
- `FetchCommentsEdge`, `FetchThreadsEdge`, `FetchEdge` accept step-level timeouts.

### Verification
- Timeouts enforced across unit and integration tests.

---

## Issue 9: Deprecation of per-edge agent field

### Problem
Configuration schemas historically contained an `agent` attribute on edges, which was superseded by Executor-level injection or self-owning edge agents.

### Solution
Removed legacy `agent` parameters from documentation and builder APIs.

---

## Issue 10: Standalone Single-Edge Driver (`run_edge`)

### Problem
Developers needed a way to test a single edge script in isolation without instantiating an entire graph.

### Solution
`framework/utils/run_edge.py`: Standalone CLI driver to load `file.py:ClassName`, execute `pre_process -> compute -> post_process`, and report outputs. Supports `--skip-compute` for offline deterministic verification.

### Changes
- `framework/utils/run_edge.py`: CLI driver and helper routines.
- `tests/test_run_edge.py`: 10 regression tests verifying resolution, skip-compute, and error handling.

### Verification
```bash
# Offline, deterministic (skips LLM call):
python -m framework.utils.run_edge --dir examples/s1_ai_report_map \
  --script s1_edges.py:SummarizeEdge --data '{"title":"T","url":"U","content":"..."}' \
  --skip-compute

# Real LLM endpoint:
python -m framework.utils.run_edge --dir examples/hn_ai_report \
  --script hn_edges.py:SummarizeEdge --data '{...}' \
  --base-url https://api.openai.com/v1/chat/completions \
  --api-key "$OPENAI_API_KEY"
```

All 353 test cases pass consistently.
