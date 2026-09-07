# OpenCode Zen — Subprocess Delegation to Local `opencode` CLI

> Documented following the "Problem / Solution / Changes / Verification" format: demonstrates using a local `opencode` CLI tool as an inference backend.

---

## Problem

The framework's `HttpLLMAgent` connects directly to HTTP endpoints; however, workflows requiring local agent loop evaluation, web search tools, or CLI workspace contexts need to delegate tasks to a **local CLI child process**.

## Solution

`OpenCodeAgentRunner`: Delegates execution to an `opencode run` child process. The edge is loaded via `script: zen_edge.py:OpenCodeEdge`, where the edge owns its `OpenCodeAgentRunner` directly in `__init__`.

## Changes

- `examples/opencode_zen/zen_edge.py`:
  ```python
  class OpenCodeEdge(Edge):
      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.agent = OpenCodeAgentRunner()
  ```
- `examples/opencode_zen/config.json`: Configures `script: zen_edge.py:OpenCodeEdge` with prompt and model settings.
- `examples/opencode_zen/run.py`: Loads the graph and executes it via `Executor(graph)`.

## Verification

- **Test Plan**: Verify the `opencode` CLI subprocess is invoked and its output flows to downstream vertices.
- **Method**:
  ```bash
  python examples/opencode_zen/run.py
  ```
  (Requires `opencode` CLI in PATH.)
- **Result**: `prompt_in -- e_zen --> zen_out` pipeline executes cleanly, with `zen_out` receiving CLI output.

### Related Examples
- `examples/real_pi/`: Subprocess delegation to local `pi` CLI (`PiAgentRunner`).
- `examples/opencode_zen/proxy_demo.py`: Transport proxy demonstration for HTTP agents.
