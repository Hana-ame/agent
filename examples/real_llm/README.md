# Real LLM Endpoint Example

This example sends a request to a real external LLM provider — [OpenCode Zen](https://opencode.ai/zen/v1) (`hy3-free`, free, no key) — instead of the test `MockAgent`. The HTTP request goes out **through a transport HTTP proxy set directly in `config.json`** (here: `https_proxy: http://127.0.1.6:7890`).

## How it works

The edge `e_real_llm` in `config.json` declares its own agent in `settings.agent`; the framework resolves it via `get_agent()` and `Edge.compute` picks it up (per-edge agent > executor-level agent). No custom `Edge.execute()` subclass is needed:

```jsonc
{ "id": "e_real_llm", "source": "user_input", "destination": "llm_output",
  "settings": {
    "prompt": "You are a creative poet.",
    "model": "hy3-free",
    "agent": {
      "type": "http",
      "base_url": "https://opencode.ai/zen/v1",
      "https_proxy": "http://127.0.1.6:7890"   // ← HTTP 请求经这个代理出去
    }
  } }
```

The `settings.agent` block accepts `https_proxy` / `proxy` / `HTTPS_PROXY` — all mean the transport HTTP(S)/SOCKS proxy. Setting it in graph.json **overrides** any `HTTP_PROXY` / `HTTPS_PROXY` from the environment.

## Execution

```bash
python examples/run.py examples/real_llm/config.json
```

Swap `127.0.1.6:7890` for one of `127.0.{1,2,3}.{4,6}:7890` (a local Clash-style egress proxy) or your own proxy endpoint. If you drop the `https_proxy` key, the agent falls back to the environment's `HTTP_PROXY` / `HTTPS_PROXY` (via `trust_env`, default on).

> **Note on the old `llm_edge.py` style**: an older version of this example overrode `Edge.execute()` and used `urllib.request.urlopen` by hand. `urllib` only honours proxy *environment variables* — it cannot pin a proxy from the graph config or override the environment. The current example uses the built-in `HttpLLMAgent` instead, which supports both.