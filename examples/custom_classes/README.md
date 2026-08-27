# Native Subclassing Example

This example demonstrates the true Object-Oriented power of the framework. Instead of using module-level hooks (`on_receive`), it defines native subclasses of `Vertex` and `Edge` that are dynamically loaded and instantiated.

## How it works

1. `config.json` specifies `"script": "my_nodes.py"` for `filter_node` and `e_smart`.
2. The framework automatically parses `my_nodes.py` using the `inspect` module to locate classes that inherit from `Vertex` and `Edge`.
3. It natively constructs `SafeFilterVertex` and `PrefixEdge`, passing in the JSON configurations.
4. Method overrides (such as `set()`, `pre_process()`, and `post_process()`) seamlessly integrate into the executor loop.

## Execution

Run the unified runner pointing to this directory's `config.json`:

```bash
python examples/run.py examples/custom_classes/config.json
```
