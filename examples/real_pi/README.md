# Real Pi — Subprocess Delegation to Local `pi` CLI

> Documented following the "Problem / Solution / Changes / Verification" format: demonstrates using a local `pi` CLI tool as an inference backend via child subprocesses.

---

## Problem

While `real_llm` uses `HttpLLMAgent` for direct HTTP endpoints, environments standardized on the local `pi` CLI (AgentCLI) require delegating inference to subprocesses rather than remote network requests.

## Solution

`PiAgentRunner`: Delegates inference to a `pi -p --model ... --system-prompt ... -- <data>` child process. The edge is loaded via `script: pi_edge.py:PiEdge`, holding its own `PiAgentRunner` directly in `__init__` without external agent injection.

## Changes

- `examples/real_pi/pi_edge.py`:
  ```python
  class PiEdge(Edge):
      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.agent = PiAgentRunner(...)
  ```
- `examples/real_pi/config.json`: Configures `script: pi_edge.py:PiEdge` and declares prompt/model in `settings`.
- Executed via `examples/run.py` loading the graph into `Executor(graph)`.

## Verification

- **Test Plan**: Verify the `pi` CLI subprocess executes and its stdout becomes the edge result.
- **Method**:
  ```bash
  python examples/run.py examples/real_pi/config.json
  ```
  (Requires `pi` CLI in system PATH.)
- **Result**: Execution pipeline `user_input -- e_real_pi (PiAgentRunner) --> pi_output` delivers the CLI output to `pi_output`; failures emit `EdgeSignal.FAILED` and propagate errors cleanly.

> Follows the same subprocess delegation architecture as `opencode_zen` (`OpenCodeAgentRunner`).
