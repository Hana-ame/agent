# SenseNova Free LLM (no proxy)

Clone of `real_llm` for the **free** [SenseNova](https://token.sensenova.cn) endpoint — model `sensenova-6.8-flash-lite`, **directly reachable without any proxy**.

## How it works

The edge `e_sensenova` in `config.json` is loaded by `script: sensenova_edge.py:SensenovaEdge`. The edge owns its `HttpLLMAgent`; the base URL / model come from `settings`, the API key from the `SENSENOVA_API_KEY` env var:

```jsonc
{ "id": "e_sensenova", "source": "user_input", "destination": "sensenova_output",
  "channel": "text", "concurrency_type": "llm",
  "script": "sensenova_edge.py:SensenovaEdge",
  "settings": {
    "prompt": "You are a creative poet.",
    "model": "sensenova-6.8-flash-lite",
    "base_url": "https://token.sensenova.cn/v1"
  } }
```

## Run

```bash
export SENSENOVA_API_KEY=sk-...   # or it reads ~/.config/opencode/opencode.json
python examples/run.py examples/sensenova/config.json
```

No `HTTPS_PROXY` needed — SenseNova is directly reachable. Nothing is injected by the runner and no fallback default agent is used.
