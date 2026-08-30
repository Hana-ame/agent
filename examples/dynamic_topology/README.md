# Dynamic Topology — 运行时图增长

> 按「问题 / 方案 / 修改 / 测试」记录：解决「图跑的时候按需生成 worker 顶点」。v2.0 特性。

## 问题

固定拓扑无法表达「任务数运行时才知道」的场景：Manager 发任务列表，每个任务要一个 worker 处理，总数不定。

## 方案

Manager 顶点在运行中发出任务，框架在**执行期动态增减 worker 顶点**，每个任务一个 worker；异步 hook 承担 I/O 密集步骤。

## 修改

- `examples/dynamic_topology/demo.py`（已核实存在）。
- 依赖 `Edge` 原生 `async def` hook 与运行时图变更（框架 v2.0 特性）。

## 测试

**测试方案**：任务列表逐项生成 worker 并完成。**测试方法**：
```bash
python examples/dynamic_topology/demo.py
```
**测试结果**：Manager 发 N 任务 → N 个 worker 并行处理 → 全部回收到 sink；运行中图结构增长正常。