# Custom Classes — Native Subclasses (Loaded via `script`)

> Documented following the "Problem / Solution / Changes / Verification" format: demonstrates subclassing native Vertex/Edge classes instead of top-level hook functions.

---

## Problem

Early versions of the framework supported top-level hook functions exported from external scripts (`on_receive`, `pre_process`, etc.). That approach is deprecated: `load_class_from_script` now strictly looks for subclasses, falling back to base classes with a warning if none are found. Custom logic written in top-level functions will not execute.

## Solution

Define custom behavior by subclassing **`Vertex` or `Edge`** in external Python files. Reference them via `script` in the configuration. The framework dynamically imports and instantiates the subclass, executing methods overridden in the subclass (`on_receive`, `on_ready`, `pre_process`, `post_process`).

## Changes

- `examples/custom_classes/my_nodes.py`:
  - `SafeFilterVertex(Vertex)`: Overrides `on_receive` to validate non-empty payloads and unescape HTML entities.
  - `PrefixEdge(Edge)`: Overrides `pre_process` to prepend `[PRE]` and `post_process` to append `[POST]`.
- `examples/custom_classes/config.json`:
  - `filter_node`: `"script": "my_nodes.py"` (automatically discovers single `Vertex` subclass).
  - `e_custom`: `"script": "my_nodes.py"` (automatically discovers single `Edge` subclass).

## Verification

- **Test Plan**: Verify subclasses in external scripts are dynamically loaded, instantiated, and custom hook overrides run.
- **Method**:
  ```bash
  python examples/run.py examples/custom_classes/config.json
  ```
- **Result**:
  - `SafeFilterVertex.on_receive` executes: empty data is rejected, valid text is cleaned and ingested.
  - `PrefixEdge.pre_process` and `post_process` execute: payloads are tagged with `[PRE]...[POST]`.
  - If no subclass is found, a warning `[ScriptLoader] ... no X subclass found, falling back to X` is logged (not triggered here, clean loading).

> For files with multiple subclasses, specify the class explicitly via `script: file.py:ClassName` (e.g. as shown in `s1_ai_report_map` / `hn_ai_report`).
