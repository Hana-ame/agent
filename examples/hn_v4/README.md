# ⚡ Hacker News AI & Tech Digest (V4 Graph Architecture Demo)

A comprehensive, production-grade reference example demonstrating the full architectural power of the **Vertex-Edge Agent Framework V4.0**.

This example builds an autonomous intelligence pipeline that concurrently queries live Hacker News discussions and host telemetry, synchronizes them through a fan-in barrier, self-heals transient network failures, and emits real-time execution events.

---

## 🏗️ Graph Topology

```
                  ┌──────────────────────────────────────────────────┐
                  │              v_trigger [start]                   │
                  │   content: {"topic": "AI, Systems", "limit": 6}  │
                  └─────────┬──────────────────────────────┬─────────┘
                            │                              │
             (e_fetch_top)  │                              │ (e_sys_probe)
                            ▼                              ▼
                 ┌────────────────────┐          ┌────────────────────┐
                 │   v_raw_stories    │          │     v_sys_env      │
                 └─────────┬──────────┘          └─────────┬──────────┘
             (reject) ↺    │                               │
       [e_recover_fetch]   │ (e_filter)                    │
                           ▼                               │
                 ┌────────────────────┐                    │
                 │ v_filtered_stories │                    │
                 └─────────┬──────────┘                    │
                           │ (e_comments)                  │
                           ▼                               │
                 ┌────────────────────┐                    │
                 │   v_discussions    │                    │
                 └─────────┬──────────┘                    │
                           │ (e_merge_hn)                  │ (e_merge_sys)
                           │                               │
                           ▼                               ▼
                 ┌────────────────────────────────────────────────────┐
                 │       v_context_bundle [JSON_MERGE Barrier]        │
                 │   {"hn_stories": [...], "system_telemetry": {...}} │
                 └─────────────────────────┬──────────────────────────┘
                                           │ (e_report)
                                           ▼
                                 ┌───────────────────┐
                                 │  v_final_report   │
                                 │   [end, report]   │
                                 └───────────────────┘
```

---

## 🌟 Core V4 Capabilities Demonstrated

| Feature | Description | In-Demo Implementation |
| :--- | :--- | :--- |
| **Two-Sided Handshake** | Strict state verification (`required_source_state`, `expected_target_state`) preventing race conditions and stale execution. | Every edge validates source and target vertex states before executing. |
| **Multi-Branch Fan-In** | Multiple concurrent edges converge into a single vertex with barrier synchronization. | `e_merge_hn` and `e_merge_sys` converge into `v_context_bundle` with `MergeStrategyV4.JSON_MERGE`. |
| **Self-Healing Reflexive Edge** | Dynamic vertex state recovery without pipeline restart. | `ReflexiveEdgeV4` on `v_raw_stories` restores curated fallback data if the live API rejects or times out. |
| **Declarative ToolEdge** | Static tool edge generating OpenAI-compatible function calls for external sandboxes. | `ToolEdgeV4` generates `bash({"command": "git log -1 --oneline"})` for harness stepping. |
| **Real-Time Event Streaming** | Asynchronous generator streaming lifecycle events as edges start and complete. | `async for event in executor.stream():` displays live pipeline progress. |
| **Edge Metrics Telemetry** | SQLite-persisted benchmarks recording execution latency, token counts, and error rates. | Stored in `edge_metrics` table, queried via `store.get_edge_metrics_summary()`. |
| **Session Staging Scratchpad** | Edge-tagged key-value persistence for pipeline diagnostics and intermediate results. | Stored in `session_staging` table, queried via `store.get_staged()`. |
| **Agent Harness Stepping** | Request-driven clock pulses for OpenAI-compatible agents (`finish_reason="tool_calls"`). | `HttpHarnessExecutorV4.step()` handles multi-turn tool interaction. |

---

## 🚀 Running the Demo

Execute the demo script directly with Python:

```bash
python3 examples/hn_v4/run.py
```

### Execution Output Breakdown:
1. **Part 1: Multi-Branch Fan-In Streaming**: Runs the concurrent graph with real-time `[STARTED]` and `[COMPLETED]` events.
2. **Part 2: SQLite Performance Benchmarks**: Prints per-edge execution latencies and staging scratchpad records.
3. **Part 3: Reflexive Self-Healing Handshake**: Deliberately injects a `reject` state into a vertex and demonstrates autonomous recovery to `data ready`.
4. **Part 4: Agent Harness Clock Stepping**: Demonstrates request pulses yielding tool calls, external harness execution, and final synthesis.

---

## 📁 File Structure

- `manifest.json`: Declarative JSON definition of vertices, edges, merge strategies, and recovery routes.
- `hn_transforms.py`: Python transformation functions, async HTTP requests to Hacker News Firebase API, system telemetry probe, and Markdown report builder.
- `run.py`: Multi-stage executable runner demonstrating both `ExecutorV4` and `HttpHarnessExecutorV4`.
- `report.md`: Generated executive Markdown digest containing curated stories, community perspectives, and system telemetry.
