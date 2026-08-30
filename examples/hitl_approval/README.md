# HITL Approval — 人工审批 + SQLite 检查点

> 按「问题/方案/修改/测试」记录：解决「敏感节点暂停等人工审批，崩溃后从快照恢复」。v2.0 特性。

## 问题

生产工作流常在敏感操作（付款/发布/高危命令）前需要**人工审批**；且长任务崩溃后需能恢复。旧框架无暂停/快照机制。

## 方案

- 顶点 `settings.require_approval: true`（或代码 `pause_for_approval()`）→ 节点进入 `PAUSED`，执行暂停；
- `CheckpointedExecutor` 在每次顶点结算后写 `SQLiteStateStore` 快照；
- 人工 `approve()` 后恢复运行；崩溃后 `resume()` 从快照续跑。

## 修改

- `framework/vertex.py`：`PAUSED` 状态 + `pause_for_approval()`/`approve()`（已核实存在）。
- `framework/executor/checkpoint.py`：`CheckpointedExecutor` + `SQLiteStateStore`（已核实存在）。
- `examples/hitl_approval/demo.py`（已核实存在）。

## 测试

**测试方案**：暂停位置正确、快照持久化、恢复后续跑。**测试方法**：
```bash
python examples/hitl_approval/demo.py
```
**测试结果**：敏感节点进入 `PAUSED`，演示路径 `approve()` 后完成；`tests/test_checkpoint.py` 覆盖快照/恢复/预审批顶点。