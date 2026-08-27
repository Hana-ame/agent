# Conditional Routing & Branch Pruning Example

This example demonstrates how to utilize the framework's built-in Edge Guard capabilities to achieve conditional data distribution, branch pruning, and cascading abort mechanisms.

## Topology

```
                   /-- [Edge Guard: intent == image] --> ImageProcessingVertex --\\
UserPromptVertex                                                                  --> ResponseCollectorVertex
                   \\-- [Edge Guard: intent == code]  --> CodeProcessingVertex  --/
```

## How It Works

1. **Edge Guard Interception**:
   The `gate_to_image` and `gate_to_code` edges will extract the data and use the built-in `evaluate_condition` to compare against `settings.match` to determine if the data matches their filter conditions.
2. **Conditional Activation & Pruning**:
   - For input data `intent: "code_generation"`, the condition for the `gate_to_image` edge is not met, so it immediately produces an `ABORTED` signal, pruning this branch.
   - At the same time, the `gate_to_code` edge condition is met, acting as a pass-through edge to transparently pass the data to `CodeProcessingVertex`.
3. **Deadlock-Free Downstream Synchronization**:
   - Upon receiving the `ABORTED` signal, `ImageProcessingVertex` transitions to the `ABORTED` state since it has no valid inputs, and continues to propagate the abort signal to the downstream `image_to_sink` edge (this is the cascading abort).
   - `ResponseCollectorVertex` (the sink node) monitors the status of all incoming edges through its internal settlement barrier. When it detects that `image_to_sink` is aborted and `code_to_sink` successfully arrives, the settlement conditions are met (all branches have resolved, and at least one succeeded). Consequently, it immediately enters `READY` and completes the graph execution. This perfectly prevents global deadlocks caused by unexecuted branches!

## Run Example

```bash
python examples/run.py examples/conditional_routing/config.json
```
