# Real LLM Endpoint Example

Sends a request to a real external LLM provider — [OpenCode Zen](https://opencode.ai/zen/v1) (`hy3-free`, free, no key) — instead of the test `MockAgent`. The HTTP request goes out **through a transport HTTP proxy** (`https_proxy` in the edge settings).

## How it works

The edge `e_real_llm` in `config.json` is loaded by `script: llm_edge.py:HttpLLMEdge`. The edge owns its `HttpLLMAgent` in Python; the base URL and the transport proxy are read from `settings`:

```jsonc
{ "id": "e_real_llm", "source": "user_input", "destination": "llm_output",
  "channel": "text", "concurrency_type": "llm",
  "script": "llm_edge.py:HttpLLMEdge",
  "settings": {
    "prompt": "You are a creative poet.",
    "model": "hy3-free",
    "base_url": "https://opencode.ai/zen/v1",
    "https_proxy": "http://127.0.1.6:7890"   // ← HTTP 请求经这个代理出去
  } }
```

`llm_edge.py`:

```python
class HttpLLMEdge(Edge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = HttpLLMAgent(
            base_url=self.settings.get("base_url", "https://opencode.ai/zen/v1"),
            proxy=self.settings.get("https_proxy"),
        )
```

The `https_proxy` key **overrides** any `HTTP_PROXY` / `HTTPS_PROXY` from the environment. Nothing is injected by the runner and no fallback default agent is used.

## Execution

```bash
python examples/run.py examples/real_llm/config.json
```

Swap `127.0.1.6:7890` for one of `127.0.{1,2,3}.{4,6}:7890` (a local Clash-style egress proxy) or your own proxy endpoint. If you drop the `https_proxy` key, `trust_env` (default on) falls back to the environment's `HTTP_PROXY` / `HTTPS_PROXY`.
