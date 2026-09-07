# Conditional Routing — Guard Dispatch and Cascading Pruning

> Documented following the "Problem / Solution / Changes / Verification" format: demonstrates dynamic branch routing and deadlock prevention on unselected branches.

---

## Problem

Conditional routing is a common requirement in agent workflows: an input must be routed to different specialized branches based on intent or properties. The challenge is ensuring **unselected branches do not cause downstream deadlocks**—downstream join vertices must be informed that an unselected branch was skipped rather than waiting indefinitely.

Earlier approaches that silently skipped execution caused downstream vertices to remain stuck in `IDLE` forever.

## Solution

Leverage framework Guards (`settings.match` / `evaluate_condition`) combined with **Cascading Abort**:

```
UserPrompt ──gate_to_image (guard: intent==image)──▶ ImageProcessing ──image_to_sink──▶ ResponseCollector
           └─gate_to_code  (guard: intent==code)  ──▶ CodeProcessing  ──code_to_sink───┘
```

- When given `intent: "code_generation"`:
  - Guard check on `gate_to_image` evaluates to false -> emits `EdgeSignal.ABORTED`, **pruning the image branch**.
  - Guard check on `gate_to_code` evaluates to true -> forwards payload to `CodeProcessing`.
- `ImageProcessing` receives `ABORTED` -> transitions to `ABORTED` state (no viable inputs) -> propagates `ABORTED` downstream across `image_to_sink` (cascading pruning).
- `ResponseCollector` monitors incoming edges via the Settlement Barrier: `image_to_sink` is aborted, `code_to_sink` succeeded -> meets the condition "all incoming edges settled and at least one succeeded" -> transitions to READY immediately.
- **Deadlock Free**: Inactive branches explicitly signal abandonment rather than hanging silently.

## Changes

- `examples/conditional_routing/config.json`: Two gate edges + two collector edges; each gate configures `settings.match` (`{"intent": "image"}` / `{"intent": "code"}`); `ResponseCollector` acts as the join vertex.

## Verification

- **Test Plan**: Verify image branch is pruned when `intent=code`, code branch executes, collector settles, and no deadlock occurs.
- **Method**:
  ```bash
  python examples/run.py examples/conditional_routing/config.json
  ```
- **Result**: Execution logs confirm `gate_to_image -> ABORTED`, `ImageProcessing -> ABORTED`, and `image_to_sink -> ABORTED` cascade; `gate_to_code` executes -> `CodeProcessing` -> `code_to_sink` succeeds; `ResponseCollector` settles as soon as both incoming edges resolve (1 success + 1 aborted) -> graph reaches DONE with 0 timeouts.
