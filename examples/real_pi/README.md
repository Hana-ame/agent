# Real Pi Agent CLI Example

This example mirrors [`real_llm`](../real_llm) but swaps the direct HTTP
call to an OpenAI-compatible endpoint for a delegation to the real
**Pi Agent CLI** via the framework's
[`PiAgentRunner`](../../framework/agents.py).

## How it works

1. In `config.json`, the edge `e_real_pi` declares `"script": "pi_edge.py"`,
   so the graph loader replaces the default `Edge` class with `PiEdge`
   (the first `Edge` subclass found in the module).
2. `examples/run.py` detects `real_pi` in the config path and injects a
   `PiAgentRunner` instance as the executor's agent.
3. When the scheduler activates this edge, `PiEdge.execute()` bypasses
   the built-in 5-stage pipeline and instead:
   1. reads data from the upstream vertex via `fetch_data(channel)`,
   2. calls `agents.process(data, prompt, model, settings)` — which
      `PiAgentRunner` turns into a
      `pi -p --model <model> --system-prompt <prompt> -- <data>`
      subprocess,
   3. writes the CLI's output to the downstream vertex via
      `receive_signal(EdgeSignal.COMPLETED)`.
4. On failure it emits `EdgeSignal.FAILED` and re-raises.

Unlike `real_llm` — which inlines a raw `urllib` HTTP POST and ignores
the injected agent — `real_pi` deliberately reuses the framework's
`PiAgentRunner` through the standard `agents` parameter, so the agent
selection lives entirely in `run.py`.

## Prerequisites

The `pi` CLI must be installed and on your `PATH`:

```bash
npm install -g @earendil-works/pi-coding-agent
```

Verify with:

```bash
pi --version
```

## Execution

Use the unified execution script pointing to the `config.json` in this
directory:

```bash
python examples/run.py examples/real_pi/config.json
```

The sink vertex `pi_output` will receive the Pi CLI's response.
