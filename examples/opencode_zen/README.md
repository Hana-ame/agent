# OpenCode Zen + Proxied LLM

Two real-LLM agents in one graph, both wired **declaratively** from `config.json`:

```mermaid
graph LR
  prompt_in((prompt_in)) -- "e_zen: OpenCodeAgent" --> zen_out((zen_out))
  prompt_in -- "e_proxy: ProxiedLLMAgent" --> proxy_out((proxy_out))
```

- **`e_zen`** calls [OpenCode Zen](https://opencode.ai/zen/v1) directly — free models, no API key, and **self-throttled** (`max_concurrency=2`, `requests_per_minute=20`) so a wide graph queues locally instead of getting 429'd.
- **`e_proxy`** routes through a self-hosted gateway (LiteLLM / one-api / an internal sidecar) and rewrites the graph-level alias `"cheap"` → `"deepseek-v4-flash"` via `model_map`.

## Run

```bash
# The OpenCode Zen edge works out of the box (free, no key).
python examples/opencode_zen/run.py
```

For the proxy edge, point the agent at a gateway that speaks OpenAI `chat/completions`:

```bash
export LLM_PROXY_BASE_URL=http://localhost:8000/v1   # LiteLLM / one-api / ...
export LLM_PROXY_API_KEY=sk-...
python examples/opencode_zen/run.py
```

### HTTP 请求经代理出去 (transport proxy)

All three HTTP agents also accept a transport-level `proxy` — the HTTP(S)/SOCKS proxy the request *tunnels through* on its way to the endpoint, independent of the gateway above:

```jsonc
"agent": { "type": "http", "proxy": "http://user:pass@corp-proxy:3128" }
"agent": { "type": "opencode", "proxy": "socks5://host:1080" }
```

When `proxy` is unset, `trust_env` (default `true`) reads `HTTP_PROXY` / `HTTPS_PROXY` from the environment.


## How the per-edge agent wiring works

Each edge's `settings.agent` is a config block resolved by `get_agent()`:

```jsonc
"agent": { "type": "opencode", "max_concurrency": 2, "requests_per_minute": 20.0 }
"agent": { "type": "proxy",    "proxy_url": "...",  "model_map": {"cheap": "deepseek-v4-flash"} }
```

`Edge.compute` picks the agent with **most-specific-wins** precedence: per-edge `settings.agent` → executor-level agent → `MockAgent`. So `run.py` passes `agents=None` and every edge still speaks to its own declared endpoint. See the **Agent Engines** section of the top-level `README.md` for the full knob reference.
