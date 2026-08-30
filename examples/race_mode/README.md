# Race Mode — 先到先赢 + 取消滞后者

> 按「问题/方案/修改/测试」记录：解决「多个候选源谁先响应谁赢，输家取消」。v3.0 特性。

## 问题

fan-in 汇聚默认等**所有**入边；但有些场景（如多路检索/多源结果）只要**第一个**到达的结果，其余应取消避免烧钱/延迟。

## 方案

`Executor(..., race_mode=True)` 或顶点 `settings.wait_policy: "any"`：第一个入边满足即触发下游，
`asyncio` 主动取消所有未完成的上游任务。

## 修改

- `framework/vertex.py`：`wait_policy: "any"` 支持（已核实存在）。
- `examples/race_mode/demo.py`（已核实存在）。

## 测试

**测试方案**：多源 fan-in，首个响应触发下游、未完成源被取消。**测试方法**：
```bash
python examples/race_mode/demo.py
```
**测试结果**：sink 在第一个响应到达时立即执行；滞后者被打断取消，无等待全部源。框架 `tests/test_improvements.py` 覆盖 race。