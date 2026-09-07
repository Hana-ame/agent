# Real LLM — Real Endpoints with Transport Proxies

> Documented following the "Problem / Solution / Changes / Verification" format: demonstrates dispatching real LLM requests through HTTP(S) transport proxies.

---

## Problem

The framework's default `MockAgent` is strictly for testing; production workflows require calls to live LLM endpoints. Furthermore, deployment environments frequently mandate routing requests through **transport proxies** (corporate egress, local forwarders, etc.). Relying on ambient `HTTPS_PROXY` environment variables can be error-prone and hard to reproduce across environments.

## Solution

- Load the edge via `script: llm_edge.py:HttpLLMEdge`, where the edge instantiates and maintains its own `HttpLLMAgent` in `__init__`.
- Explicitly declare `base_url` (full URL including path) and `https_proxy` within `settings`. Configuration values take explicit precedence over ambient `HTTP_PROXY` and `HTTPS_PROXY` environment variables.

## Changes

- `examples/real_llm/llm_edge.py`:
  ```python
  class HttpLLMEdge(Edge):
      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.agent = HttpLLMAgent(
              base_url=self.settings.get("base_url", "https://opencode.ai/zen/v1"),
              proxy=self.settings.get("https_proxy"),
          )
  ```
- `examples/real_llm/config.json`: Configures `settings.base_url` and `settings.https_proxy` (e.g. `http://127.0.1.6:7890` or custom local forwarder).

## Verification

- **Test Plan**: Verify `https_proxy` overrides environment variables and successfully queries the remote LLM endpoint.
- **Method**:
  ```bash
  python examples/run.py examples/real_llm/config.json
  ```
  (Set proxy endpoint according to your local environment.)
- **Result**: Data pipeline `user_input -- e_real_llm (HttpLLMAgent) --> llm_output` completes, with `llm_output` receiving the live response; when `https_proxy` is omitted, `trust_env=True` falls back cleanly to environment variables.
