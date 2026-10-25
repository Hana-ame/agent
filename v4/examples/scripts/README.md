# Scripts — Shared Subclass Definitions

> Referenced via `script` by examples such as `examples/complex` and `examples/custom_classes`. All implementations are **subclasses** rather than deprecated top-level hook functions.

## Problem

Multiple example pipelines require shared data processing logic (case transformation, prefix/suffix tagging, input validation). This logic should be shared in reusable modules rather than duplicated.

## Solution

Place reusable subclasses in a common directory and reference them via relative paths in `script` (resolved relative to the configuration file).

## Changes

- `uppercase_handler.py`: Subclasses `UpperVertex(Vertex)` to uppercase payloads in `on_receive` and aggregate in `on_ready`.
- `prefix_handler.py`: Subclasses `PrefixEdge(Edge)` to apply prefixes in `pre_process` and suffixes in `post_process` (configurable via `settings.prefix` / `settings.suffix`).
- `validator.py`: Subclasses `ValidatorVertex(Vertex)` to validate inputs and reject malformed data.

## Verification

- **Test Plan**: Verify subclasses are dynamically loaded and executed across multiple examples.
- **Method**:
  ```bash
  python examples/run.py examples/complex/config.json
  ```
- **Result**: Transformations and validations execute successfully; covered in `tests/test_script_loader.py`.
