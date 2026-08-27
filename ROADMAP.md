# Vertex-Edge Agent Framework: Development Roadmap

## ✅ v1.0: Core Architecture (Completed)
- **Event-Driven DAG Engine**: Non-interactive JSON-driven execution.
- **Unified 5-Stage Edge Pipeline**: Guard -> Pre-Process -> Compute -> Post-Process -> Deliver.
- **Unified Message Passing**: Edge-Vertex communication consolidated into `handle_edge_signal` using `EdgeSignal`.
- **Dynamic Branching & Pruning**: Deadlock-free conditional routing using Edge Guards and Cascading Aborts.
- **Extensibility**: Native subclassing and script hooks (`on_receive`, `on_ready`, `pre_process`, `post_process`).
- **Concurrency**: Semaphore-bounded asynchronous execution.

## 🚀 v2.0: Application-Ready (Next Steps)
1. **Retry Mechanism & Exponential Backoff**: Add resilience to LLM API rate limits and network instability.
2. **State Persistence & Checkpointing**: Enable pausing workflows, saving state snapshots, and Human-in-the-Loop (HITL) interventions.
3. **Real-Time Event Streaming**: Provide async generators for real-time observability, enabling SSE (Server-Sent Events) or WebSockets for frontend clients.
4. **Stateful Loops & Cycles**: Evolve from a strict DAG to support stateful loops and self-correction cycles with bounded iteration limits and state resets.

## 🌌 v3.0: Enterprise-Grade (Future Vision)
1. **Nested Sub-Graphs**: Allow vertices to encapsulate entire independent graphs for scalable multi-agent teamwork.
2. **Global Memory & Context Management**: Implement a decoupled `MemoryStore` for long-term/short-term context, preventing token window bloat across sequential steps.
3. **Telemetry & Tracing**: Integrate OpenTelemetry/LangSmith for granular tracking of token usage, API costs, and edge latency.
4. **Distributed Execution**: Decouple `Executor` from `Graph` via message queues (e.g., Redis, RabbitMQ) to allow multi-node worker clusters for heavy workloads.
