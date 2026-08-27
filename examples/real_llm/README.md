# Real LLM Endpoint Example

This example demonstrates how to completely bypass the framework's mock `PIAgent` by overriding the `Edge.execute()` method in a custom Edge subclass.

Instead of just modifying the prompt or the output, the `RealLLMEdge` class makes an actual HTTP POST request to an OpenAI-compatible endpoint (`https://opencode.ai/zen/v1/chat/completions`) using the model specified in `config.json`.

## How it works

1. `config.json` specifies `"script": "llm_edge.py"` for the edge `e_real_llm`.
2. The framework loads `llm_edge.py` and dynamically instantiates `RealLLMEdge` in place of the base `Edge` class.
3. When the executor fires the edge, it calls our custom `execute()` method.
4. `execute()` fetches the data from the source vertex, builds the JSON payload, calls the external API asynchronously, parses the response, and writes it directly to the destination vertex.

## Execution

Run this example using the unified runner:

```bash
python examples/run.py examples/real_llm/config.json
```
