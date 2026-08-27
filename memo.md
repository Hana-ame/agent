# Project Memo

## 2026-08-27
- Applied dynamic subclass support for Vertex and Edge in graph.py via inspect module.
- Conducted architectural review of the framework, identifying polling vs event-driven issues, error propagation flaws, and input tracking mechanisms.
- Executed full framework refactoring resolving polling vs event-driven issues (now uses asyncio.wait), error propagation flaws (mark_edge_failed), and input tracking (completed_incoming_edges sets). All 62 pytest cases pass. Posted v1.1 to Moonchan.
- Merged `GateEdge` logic into `Edge` to create a unified 5-stage Edge pipeline (Guard -> Pre-Process -> Compute -> Post-Process -> Deliver).
- Merged vertex and edge communication methods (`get`, `set`, `mark_edge_failed`, `mark_edge_aborted`) into a single `handle_edge_signal` method using `EdgeSignal` enum (Message-Passing architecture).
- Updated all tests (now 72/72 passing) and translated all README documentation to Chinese.
