# Self-Correction — Business Retries with Feedback Loops

> Documented following the "Problem / Solution / Changes / Verification" format: demonstrates automatic error feedback injection and retry loops for LLM validation errors. Framework v2.0 feature.

## Problem

LLM responses frequently fail structural or semantic constraints (e.g. malformed JSON or missing required fields). Halting execution requires manual intervention, while naive retry loops risk polluting the prompt context.

## Solution

Configure `settings.retry_policy`: `{"max_retries": N, "backoff_factor": x, "retry_on": [...]}`.
When `post_process` raises an exception, the framework captures the error, injects the traceback into the prompt (`[SYSTEM FEEDBACK: ...]`), and retries with exponential backoff.
The framework freezes `_base_prompt` and reconstructs an ephemeral `active_prompt` per retry attempt to prevent feedback block accumulation across iterations.

## Changes

- `framework/edge.py`: Integrated retry policy, `_base_prompt` isolation, and post-process error feedback.
- `examples/self_correction/demo.py`: Demonstrates handling corrupted outputs and retrying with feedback.

## Verification

- **Test Plan**: Verify post-processing errors trigger feedback injection, exponential backoff, and prompt cleanup.
- **Method**:
  ```bash
  python examples/self_correction/demo.py
  pytest tests/test_retry_and_stream.py -q
  ```
- **Result**: Initial corrupted LLM output automatically triggers feedback-assisted retry until valid output is produced; regression tests assert exactly one feedback block is retained.
