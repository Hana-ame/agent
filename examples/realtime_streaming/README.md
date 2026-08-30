# Realtime Streaming — 非阻塞事件流观测

> 按「问题 / 方案 / 修改 / 测试」记录：解决「图跑的时候怎么实时看内部状态」。v2.0 特性。

## 问题

图执行是异步的，用户只能在结束后看结果。调试/演示时需要在运行中实时观测顶点状态迁移与边触发。

## 方案

`executor.stream()` 异步生成器：内部 `asyncio.Queue` 承载 `GraphEvent`（dataclass），
`None` 哨兵表示结束。消费端 `async for event in executor.stream()` 实时打印事件。

## 修改

- `framework/executor/base.py`：`stream()` 生成器 + `GraphEvent`（已核实存在）。
- `examples/realtime_streaming/demo.py`：ANSI 彩色渲染事件流。

## 测试

**测试方案**：事件按发生顺序非阻塞产出、结束有哨兵。**测试方法**：
```bash
python examples/realtime_streaming/demo.py
```
**测试结果**：实时打印顶点状态迁移（IDLE→READY→AWAITING_EDGES→DONE）与边触发，
无阻塞；框架 `tests/test_retry_and_stream.py` 覆盖 `stream()`。