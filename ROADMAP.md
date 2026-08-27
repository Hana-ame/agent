# Vertex-Edge Agent Framework: Development Roadmap

## ✅ v1.0: Core Architecture (Completed)
- **Event-Driven DAG Engine**: Non-interactive JSON-driven execution.
- **Unified 5-Stage Edge Pipeline**: Guard -> Pre-Process -> Compute -> Post-Process -> Deliver.
- **Unified Message Passing**: Edge-Vertex communication consolidated into `handle_edge_signal` using `EdgeSignal`.
- **Dynamic Branching & Pruning**: Deadlock-free conditional routing using Edge Guards and Cascading Aborts.
- **Extensibility**: Native subclassing and script hooks (`on_receive`, `on_ready`, `pre_process`, `post_process`).
- **Concurrency**: Semaphore-bounded asynchronous execution.

## ✅ v2.0: Application-Ready & Interactive Features (Completed)
1. **Business-Logic Retry & Self-Correction**: `retry_policy` support in `EdgePipeline` with exponential backoff and error prompt reflection (`[SYSTEM FEEDBACK: ...]`).
2. **State Persistence & Checkpointing**: `SQLiteStateStore` snapshot storage, `CheckpointedExecutor` with graph pause/resume lifecycle.
3. **Human-in-the-Loop (HITL) & Native Approvals**: Native `VertexState.PAUSED`, `pause_for_approval()`, `approve(data)`, and JSON `"require_approval": true` settings.
4. **Real-Time Event Streaming**: Async generator `executor.stream()` producing standard `GraphEvent` instances over a non-blocking sidecar queue.
5. **Stateful Loops & Cycles**: Controlled cycle validation with bounded `max_iterations`, loop-back edge routing, iteration state tracking, and re-entry scheduling.

## 🌌 v3.0: Enterprise-Grade (In Progress)
1. **✅ Nested Sub-Graphs (`SubgraphVertex`)**: Allow vertices to encapsulate entire independent graphs for scalable multi-agent teamwork, with boundary routing (`input_map`/`output_map`), event stream bubbling, and namespaced checkpoint persistence.
2. **✅ Global Memory & Shared Context (`MemoryStore`)**: Decoupled, thread-safe key-value bus supporting TTLs, sub-namespaces, and declarative edge reads/writes (`memory_read`/`memory_write`).
3. **✅ Telemetry & Cost Tracing (`TelemetryTracker`)**: Granular per-edge and workflow-level tracking of prompt tokens, completion tokens, execution latency, and model-specific USD cost estimates.
4. **Distributed Execution**: Decouple `Executor` from `Graph` via message queues (e.g., Redis, RabbitMQ) to allow multi-node worker clusters for heavy workloads.
