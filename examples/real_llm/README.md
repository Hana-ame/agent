# Real LLM Endpoint Example

This example demonstrates how to completely bypass the built-in test `MockPIAgent` by subclassing `Edge`, thereby sending requests directly to a real external LLM provider.

Unlike standard `pre_process` (which modifies the prompt) or `post_process` (which modifies the returned result), `RealLLMEdge` completely reconstructs the internal workflow. It uses the built-in `urllib` library in conjunction with `asyncio.to_thread` to make real, asynchronous HTTP POST network requests to an OpenAI-compatible endpoint (such as `https://opencode.ai/zen/v1/chat/completions`). The model used for the request is dynamically specified in `config.json`.

## How it works

1. In `config.json`, the edge `e_real_llm` is configured to use the external extension `"script": "llm_edge.py"`.
2. The framework loads `llm_edge.py` and, when building the graph, automatically replaces the default `Edge` class with the `RealLLMEdge` subclass.
3. When the scheduler (Executor) activates this edge, it does not use the built-in PI Agent. Instead, it executes the custom invocation logic we overrode in the subclass.
4. This invocation logic includes: reading data from the upstream Vertex, assembling the JSON request body, initiating a non-blocking HTTP request, parsing and extracting the LLM's response, and writing it fully to the downstream target Vertex via `handle_edge_signal`.

## Execution

Use the unified execution script pointing to the `config.json` in this directory:

```bash
python examples/run.py examples/real_llm/config.json
```
