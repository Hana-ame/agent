# SenseNova — Direct Inference Endpoint (No Proxy)

> Documented following the "Problem / Solution / Changes / Verification" format: demonstrates direct connection to the SenseNova inference endpoint without requiring transport proxies.

---

## Problem

While `real_llm` routes requests through transport proxies, public API endpoints are frequently **directly accessible** without requiring any proxy. Forcing requests through proxy environment variables can introduce latency or routing failures.

## Solution

Use `script: sensenova_edge.py:SensenovaEdge` to load a custom Edge that manages its own `HttpLLMAgent` in `__init__`. The `base_url` and `model` are configured in `settings`, and credentials are read from the `SENSENOVA_API_KEY` environment variable directly without proxy dependencies.

## Changes

- `examples/sensenova/sensenova_edge.py`: `SensenovaEdge` initializes its agent with `base_url` pointing to `https://sensenova.moonchan.xyz/v1/chat/completions` and model `sensenova-6.8-flash-lite`.
- `examples/sensenova/config.json`: Declares `script: sensenova_edge.py:SensenovaEdge` and endpoint settings.

## Verification

- **Test Plan**: Verify direct outbound connectivity and prompt completion without `HTTPS_PROXY`.
- **Method**:
  ```bash
  export SENSENOVA_API_KEY=sk-...
  python examples/run.py examples/sensenova/config.json
  ```
- **Result**: `user_input -- e_sensenova (SensenovaEdge) --> sensenova_output` executes cleanly with direct API response.
