# 动态分支路由与条件剪枝 (Conditional Routing)

本示例展示了如何利用框架内置的 Edge Guard (门限拦截) 能力，实现基于条件的数据分发、分支剪枝 (Branch Pruning) 和雪崩取消机制 (Cascading Abort)。

## 拓扑结构 (Topology)

```
                   /-- [Edge 拦截条件: intent == image] --> ImageProcessingVertex --\\
UserPromptVertex                                                                     --> ResponseCollectorVertex
                   \\-- [Edge 拦截条件: intent == code]  --> CodeProcessingVertex  --/
```

## 运行原理 (How It Works)

1. **Edge 的门卫拦截 (Guard)**:
   `gate_to_image` 和 `gate_to_code` 边会在提取到数据后，通过内置的 `evaluate_condition` 对比 `settings.match`，判断数据是否符合自己的放行条件。
2. **条件激活与剪枝 (Conditional Activation & Abort)**:
   - 对于输入数据 `intent: "code_generation"`，`gate_to_image` 边的条件不满足，立刻产生 `ABORTED` 信号，从而切断了该分支。
   - 与此同时，`gate_to_code` 边条件满足，充当透明管道 (Pass-through edge) 将数据透传给 `CodeProcessingVertex`。
3. **无死锁的并发合并 (Deadlock-Free Downstream Synchronization)**:
   - `ImageProcessingVertex` 在收到 `ABORTED` 信号后，由于没有任何有效输入，其自身也变为 `ABORTED` 状态，并继续将取消信号向下游的 `image_to_sink` 边传递（这就是雪崩取消）。
   - `ResponseCollectorVertex`（端点节点）会通过内部的结算屏障，监控所有入边的定论。当它发现 `image_to_sink` 被取消，而 `code_to_sink` 成功抵达时，结算条件满足（所有分支皆有结果且至少有一个成功），于是立刻进入 `READY` 并最终完成图的执行。完美避免了因为某个分支不执行而导致的全局死锁！

## 运行示例 (Run Example)

```bash
python examples/run.py examples/conditional_routing/config.json
```
