# Self-Correction — 业务重试 + 自纠错

> 按「问题/方案/修改/测试」记录：解决「LLM 输出格式/领域错误时自动注入反馈再试」。v2.0 特性。

## 问题

LLM 输出常不符合约束（JSON 解析失败、业务字段缺失）。旧框架失败即停，需人工介入；重试时会污染 prompt。

## 方案

`settings.retry_policy`：`{"max_retries": N, "backoff_factor": x, "retry_on": [...]}`。
`post_process` 抛错 → 捕获 → 错误堆栈注入 prompt（`[SYSTEM FEEDBACK: ...]`）→ 指数退避重试。
框架冻结 `_base_prompt`，每次重试从它重建 `active_prompt`，不跨迭代累积（commit `121ea9e`）。

## 修改

- `framework/edge.py`：retry_policy + `_base_prompt` 隔离 + `post_process` 错误注入（已落地）。
- `examples/self_correction/demo.py`（已核实存在）。

## 测试

**测试方案**：post 错误被捕捉、反馈注入、按退避重试、单 feedback 块不堆积。
**测试方法**：`python examples/self_correction/demo.py` + `pytest tests/test_retry_and_stream.py -q`。
**测试结果**：演示中 LLM 首次输出损坏 → 自动注入反馈重试至成功；回归断言仅单个 `[SYSTEM FEEDBACK]` 块，通过。