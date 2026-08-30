# Conditional Routing — Guard 条件分发 + 级联剪枝

> 按「问题 / 方案 / 修改 / 测试」记录：这个实验解决「同一数据按条件走不同分支，且失败分支不死锁」。 

---

## 问题

条件路由是 Agent 流水线常见需求：一条输入按意图分发到不同处理分支。难点是**未命中分支不能被忽略**
导致下游死等——下游节点必须知道「这条分支被放弃了」，且整图仍能正常结束。

旧示例用「单边 guard + 忽略」，条件不满足时没有任何信号，下游永久 `IDLE`。

## 方案

利用框架内置 Guard（`settings.match` / `evaluate_condition`）+ **Cascading Abort**：

```
UserPrompt ──gate_to_image (guard: intent==image)──▶ ImageProcessing ──image_to_sink──▶ ResponseCollector
           └─gate_to_code  (guard: intent==code)  ──▶ CodeProcessing  ──code_to_sink───┘
```

- 输入 `intent: "code_generation"`：
  - `gate_to_image` 的 guard 命中失败 → 立即发 `ABORTED` 信号，**剪掉 image 分支**；
  - `gate_to_code` 命中 → 透传数据到 `CodeProcessing`。
- `ImageProcessing` 收到 `ABORTED` → 进入 `ABORTED`（无有效输入）→ 继续向下游 `image_to_sink`
  传播 `ABORTED`（级联剪枝）。
- `ResponseCollector`（汇聚）通过 Settlement Barrier 统计所有入边：`image_to_sink` 已 abort、
  `code_to_sink` 已成功 → 条件「全部已解决且至少一个成功」满足 → 立即 READY，图正常结束。
- **无死锁**：失败分支用信号显式「放弃」，而不是静默缺失。

## 修改

- `examples/conditional_routing/config.json`：两条 gate 边 + 两条汇聚边；每条 gate 配
  `settings.match`（如 `{"intent": "image"}` / `{"intent": "code"}`）；`ResponseCollector` 为汇聚节点。

## 测试

**测试方案**：`intent=code` 时 image 分支被剪、仅 code 分支执行、sink 正常完成、整图无死锁。
**测试方法**：
```bash
python examples/run.py examples/conditional_routing/config.json
```
**测试结果**：日志显示 `gate_to_image -> ABORTED`、`ImageProcessing -> ABORTED`、
`image_to_sink -> ABORTED` 级联；`gate_to_code` 透传 → `CodeProcessing` → `code_to_sink` 成功；
`ResponseCollector` 在两条入边都「已解决」（1 成功 + 1 剪枝）后 READY → 全图 DONE。无等待超时。