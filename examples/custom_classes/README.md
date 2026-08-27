# Native Subclassing Example

This example demonstrates the true object-oriented programming (OOP) power of this framework. Instead of using traditional external module-level hooks (such as registering the `on_receive` function in a `.py` file), it directly defines native subclasses of `Vertex` and `Edge` in an external script, which are dynamically loaded and instantiated by the framework.

## How it works

1. In `config.json`, we configure `"script": "my_nodes.py"` for `filter_node` and `e_smart`.
2. The framework automatically scans `my_nodes.py` using Python's native `inspect` module to intelligently identify subclasses that inherit from `Vertex` and `Edge`.
3. The framework natively constructs instances of `SafeFilterVertex` and `PrefixEdge`, passing the JSON configuration directly to their `__init__` constructors.
4. Subclasses can directly define or override lifecycle methods (such as defining a native `on_receive()` method in the vertex, or overriding `pre_process()` and `post_process()` in the edge). These class methods seamlessly integrate into the Executor's event loop mechanism.

## Execution

Use the unified execution script pointing to the `config.json` in this directory:

```bash
python examples/run.py examples/custom_classes/config.json
```
