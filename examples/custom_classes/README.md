# 动态面向对象子类示例 (Native Subclassing)

本示例展示了该框架真正的面向对象 (OOP) 威力。它并没有使用传统的外部模块级钩子（如挂载 `.py` 文件里的 `on_receive` 函数），而是直接在外部脚本中定义了 `Vertex` 和 `Edge` 的原生子类，由框架动态加载并实例化。

## 运行原理 (How it works)

1. 在 `config.json` 中，我们为 `filter_node` 和 `e_smart` 配置了 `"script": "my_nodes.py"`。
2. 框架会利用 Python 原生的 `inspect` 模块自动扫描 `my_nodes.py`，智能识别出其中继承自 `Vertex` 和 `Edge` 的子类。
3. 框架会原生构造 `SafeFilterVertex` 和 `PrefixEdge` 实例，并将 JSON 配置直接透传给它们的 `__init__` 构造函数。
4. 子类可以直接定义或重写生命周期方法（例如在节点里定义原生方法 `on_receive()`，在边里重载 `pre_process()` 和 `post_process()`），这些类方法会完美融入 Executor 的事件循环机制。

## 运行示例 (Execution)

使用统一运行脚本，指向本目录的 `config.json`：

```bash
python examples/run.py examples/custom_classes/config.json
```
