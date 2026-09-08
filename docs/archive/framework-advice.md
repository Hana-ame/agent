# Vertex-Edge-Agent: Architecture Review & Advice

> This document is organized by "Problem / Solution / Changes / Verification": each issue provides the solution, actual code changes, and test results.
> Status: All confirmed issues from the review have been resolved. **355 tests passed**.

---

## Part 1: `hn_ai_report` Example — Recommendations

### Issue 1: `hn_edges.py` was previously dead code (duplicated with inline config code)

#### Problem
The previous configuration contained inline Python logic while classes defined in `hn_edges.py` were unreferenced, creating confusion with two parallel execution styles.

#### Solution
Adopt the class extension approach: reference subclasses via `script: hn_edges.py:ClassName` from config and remove inline code.

#### Changes
- `examples/hn_ai_report/config.json`: Replaced inline `action_code` with `script` pointing to `hn_edges.py` (`FetchTopStoriesEdge` / `FilterEdge` / `ProcessStoriesMap`). Each step uses explicit class names such as `script: hn_edges.py:FetchCommentsEdge`.

#### Verification
- **Test Plan**: Generate end-to-end example report.
- **Method**: Run `python examples/hn_ai_report/demo.py` (with proxy in config).
- **Result**: Successfully generated `report.md` (~99 lines) covering valid forum stories.

### Issue 2: Inline Python in JSON was unmaintainable

#### Problem
Old `action_code` strings were newline-escaped Python snippets inside JSON, preventing syntax highlighting, linting, and debugging.

#### Solution
Migrate custom logic to external `.py` script files referenced by `script`.

#### Changes
- `config.json`: Removed inline code blocks; moved custom logic into subclasses in `hn_edges.py`.

#### Verification
- **Test Plan**: Verify no inline `action_code` exists in config.
- **Method**: `grep action_code examples/hn_ai_report/config.json`.
- **Result**: 0 occurrences found.

### Issue 3: Missing error handling in network calls

#### Problem
Network fetching lacked retries or fallbacks; if the external API failed, the pipeline crashed.

#### Solution
Declare explicit `timeout` settings (default 30s) on fetch steps and isolate errors using `EdgeSignal.FAILED` so external failures do not bring down the entire graph.

#### Changes
- `examples/hn_ai_report/hn_edges.py`: Fetch routines respect `timeout` in settings; added test coverage matching `tests/test_s1_edges.py`.

#### Verification
- **Test Plan**: Verify fetch timeouts or failures do not crash the Executor.
- **Method**: `pytest tests/test_s1_edges.py -q`.
- **Result**: Passed.

### Issue 4: Example lacked a dedicated README

#### Problem
The example only had a single line in the examples root README without dedicated documentation.

#### Solution
Add dedicated documentation for `examples/hn_ai_report` (integrated into `examples/README.md` and `ai_report_notes.md`).

#### Changes
- `examples/README.md`: Added entries for `hn_ai_report`, `s1_ai_report_map`, and `sensenova`; updated `ai_report_notes.md` with architecture, configuration, and comparative analysis.

#### Verification
- **Test Plan**: Index line count matches all 19 example directories.
- **Method**: `grep -c '^| **' examples/README.md`.
- **Result**: 19.

---

## Part 2: Framework Architecture — Recommendations & Status

| # | Topic | Status | Resolution |
|---|---|---|---|
| 1 | Dual execution path in Graph and Executor | Resolved | Execution consolidated into `Executor`; `Graph` serves as pure data container (+ `to_dict/to_json`) |
| 2 | Monolithic `Vertex` class (5 inline actions) | Resolved | Refactored Vertex into a state machine container; computation moved to Edge / subclasses |
| 3 | Bare dictionary for Context | Resolved | `settings` and channel carried explicitly; `ExecutionContext` provides agents, memory, telemetry |
| 4 | Security risk from `exec()` / `eval()` | Removed | 0 occurrences of `exec()` / `eval()` in framework; replaced with subclass overrides + `edge_transform` |
| 5 | Incomplete Edge signal hierarchy | Resolved | Consolidated signals into `EdgeSignal.ABORTED` / `FAILED` + settlement barrier pruning |
| 6 | Lack of observability | Implemented | Added `executor.stream()`, `GraphEvent`, and `TelemetryTracker` |
| 7 | Weak SubGraph isolation | Implemented | `SubgraphVertex` with input/output mapping translation and event bubbling |
| 8 | HTTP client connection reuse | Fixed | `_client_for(settings)` cached per proxy; idempotent `close()` cleanup |

---

## Part 3: Confirmed Bugs & Status

### Bug 1: `GraphBuilder.vertex()` ignored custom scripts
- **Problem**: Script was stored under the wrong key (`vc["pipeline"]`) and silently discarded.
- **Resolution**: Fixed key to `vc["script"] = script`.
- **Verification**: Verified via `tests/test_improvements.py`.

### Bug 2: Edge prompt accumulated across loop iterations
- **Problem**: `retry_policy` mutated `self.prompt` in place, accumulating multiple `[SYSTEM FEEDBACK]` blocks over iterations.
- **Resolution**: Froze `_base_prompt`, rebuilt `active_prompt` per retry, and restored prompt after execution.
- **Verification**: `tests/test_retry_and_stream.py` asserts exactly one feedback block; passed.

### Bug 3: README referenced non-existent method
- **Problem**: `from_json_file()` did not exist.
- **Resolution**: Standardized documentation on `from_json()`.
- **Verification**: `grep from_json README.md` confirmed 0 occurrences of `from_json_file`.

---

## Part 4: Backlog / Future Work

| # | Item | Test Status |
|---|---|---|
| 1 | Distributed execution (ROADMAP v3 #7) | Planned |
| 2 | Dynamic topology runtime growth stress testing | Example functional; dedicated stress test pending |
| 3 | 24-hour forum fetch window misses older original posts | Known limitation (MapEdge data window) |

---

> **Conclusion**: The report examples (`hn_ai_report` / `s1_ai_report_map`) now serve as canonical demonstrations of the "class extension + explicit script class names" pattern.
> All listed framework issues have been addressed, and **355 tests passed**.
