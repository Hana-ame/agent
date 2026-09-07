# Race Mode — First-to-Finish Preemption and Cancellation

> Documented following the "Problem / Solution / Changes / Verification" format: demonstrates triggering downstream execution upon the first completing source while cancelling slower branches. Framework v3.0 feature.

## Problem

By default, fan-in vertices await **all** incoming edges before settling. In scenarios such as multi-source information retrieval or redundant fallback generation, only the first response is needed; remaining operations should be cancelled to conserve latency and compute costs.

## Solution

Configure `Executor(..., race_mode=True)` or set vertex `settings.wait_policy: "any"`. Downstream processing triggers immediately as soon as the first incoming edge arrives, and the framework cancels all remaining in-flight upstream tasks.

## Changes

- `framework/vertex.py`: Added `wait_policy: "any"` support.
- `examples/race_mode/demo.py`: Demonstrates race resolution across concurrent branches.

## Verification

- **Test Plan**: Multi-source fan-in triggers sink on first arrival and cancels pending branches.
- **Method**:
  ```bash
  python examples/race_mode/demo.py
  ```
- **Result**: Sink triggers immediately upon receiving the fastest branch response; lagging tasks are cancelled cleanly; covered in `tests/test_improvements.py`.
