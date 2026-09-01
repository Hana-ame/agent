# 🔁 Vertex-Edge Agent Framework — Review Round 2 处置记录

> 第二轮 review 发现 6 个问题,全部已修复。本文档按「问题/方案/修改/测试」四段式记录。
> **范围**:`framework/`(vertex/executor/agents/utils)+ 2 示例 + 测试。**测试**:355 tests passed ✅。

---

## 问题 1：混合输入循环静默死锁

### 问题
回边进入一个同时有「一次性非回边输入」的顶点时,图卡死。`Vertex.receive_signal` 的 loop
re-entry 分支把已结算的 `completed_incoming_edges` 全部 `clear()` 掉,只记回边——一次性 seed
输入被遗忘,且该输入永不再投递,顶点永久 IDLE,进入 `Deadlock`。且 loop 分支把回边算作
「本轮必需输入」,混合顶点第一圈就等不到回边(回边要等该顶点先跑完才产生),拓扑从第一圈
起就死锁。

### 方案
- 回边是「重入触发器」,不算本轮必需输入。
- 重入时不清空已结算的非回边输入(一次性 seed 不会重投)。
- 就绪判定统一改为:所有**非回边**输入已结算(completed 或 aborted)即 READY。

### 修改
- `framework/vertex.py`:loop re-entry 分支删除 `clear()`/`_received_input_count=0`,
  改为 `completed_incoming_edges.add(edge_id)` + 按非回边输入结算判定就绪;
  正常投递分支的 readiness 计算同步改按「非回边 required 列表」判定。
- `tests/test_loops.py`:新增 `TestMixedInputLoop`(2 个用例)。

### 测试
**测试方案**：构造 `A(seed) → X → Y → X(loop) 、X → Z` 混合拓扑,max_iterations=3/5,
验证能完整转圈、无死锁、迭代计数正确。
**测试方法**：
- `pytest tests/test_loops.py::TestMixedInputLoop -v`
- 复现脚本:旧代码下运行同拓扑,断言 `Deadlock`(回归前失败路径)。
**测试结果**：2 个新用例通过;`X.iteration_count==3/5`、X/Y/Z 均 DONE、errors 为空、
`Z` 每圈都收到投递;`test_loops.py` 全量 13 passed。

---

## 问题 2：HITL 暂停被报告为「失败」

### 问题
运行到 PAUSED 顶点(人工审批)时,`ExecutionResult.success=False`、`errors=[]`、
`summary()` 打印 `FAILED ✗`,store 状态却是 `awaiting_approval`。暂停是正常状态,却被下游
按 `success` 判断的调用方误判为失败。

### 方案
给 `ExecutionResult` 增加 `paused` 标记:执行结束若有 PAUSED 顶点则置位;
`summary()` 在暂停时显示 `PAUSED ⏸ (waiting for human approval)`,不再显示 FAILED。

### 修改
- `framework/executor/base.py`:`ExecutionResult.__init__` 加 `self.paused=False`;
  `_run_internal` 结束时 `self._result.paused = any(v.state == PAUSED)`,
  `success` 计算排除 paused;`summary()` 标题按 paused 分支。
- `tests/test_checkpoint.py`:HITL 用例补断言。

### 测试
**测试方案**：HITL 图(含 require_approval 顶点)跑第一轮暂停,检查结果对象的
paused/success/errors/summary 四者一致性。
**测试方法**：
- `pytest tests/test_checkpoint.py::TestHumanGateVertex -v`
- 探针:暂停后用 `result.summary()` 打印,核对标题与 errors。
**测试结果**：`paused=True`、`success=False`、`errors==[]`、summary 含 `PAUSED` 且不含
`FAILED`;`TestHumanGateVertex` 7 passed;全量 checkpoint 用例通过。

---

## 问题 3：子进程 agent 无超时、卡住会残留

### 问题
`PiAgentRunner` / `OpenCodeAgentRunner` 用 `proc.communicate()` 无限等待;若外层任务被
取消(executor/edge 超时),子进程不回收,残留 CLI 进程。real_pi/opencode_zen 配置未设
timeout。

### 方案
- 读 `settings["timeout"]` 用 `asyncio.wait_for` 限时(如有)。
- 超时或 `CancelledError` 时 `proc.kill()` + `await proc.wait()` 回收子进程,抛明确错误。

### 修改
- `framework/agents/pi_agent_runner.py`、`framework/agents/opencode_agent_runner.py`:
  communicate 包 `wait_for`;新增 TimeoutError/CancelledError 分支 kill+wait。

### 测试
**测试方案**：造假 `pi`/`opencode` 命令(`sleep 300`)分别测「settings.timeout 超时」与
「外部取消」两条路径,断言抛错/干净取消且无残留进程。
**测试方法**：
- 探针:临时目录放 `#!/bin/sh\nsleep 300`,改写 PATH,`PiAgentRunner.process(settings={"timeout":1})`
  断言 `RuntimeError` 含 "timed out";`OpenCodeAgentRunner` 任务 0.3s 后 `cancel()`,断言
  `CancelledError` 正常传播。
- `ps -ef | grep '[s]leep 300'` 计数残留。
**测试结果**：pi 1s 超时抛 `Pi Agent CLI timed out after 1s, killed.`;opencode 外部取消
干净通过;残留进程计数 0;全量 355 passed 不受影响。

---

## 问题 4：文档/注释残留旧说法

### 问题
三处旧叙事残留:`factory._build_from_dict` docstring 仍列 `"proxy"` 为合法 type(代码只支持
http/opencode);`examples/run.py` 注释写着「per-edge agent 字段」「自动 fallback MockAgent」;
`examples/self_correction/demo.py` docstring 用已废弃的 `EdgePipeline` 名。

### 方案
删除/改为与现状一致的说法:factory docstring 只列 http|opencode;run.py 注释改描述真实
行为;demo.py 改用 `Edge`。

### 修改
- `framework/agents/factory.py`:两处 docstring 的 `|proxy` 删除。
- `examples/run.py`:旧注释替换为「普通 Edge 含 prompt/model 时回退 MockAgent;真实
  LLM/子进程 agent 由 script Edge 自持」。
- `examples/self_correction/demo.py`:`EdgePipeline` → `Edge`,重写第一段 docstring。

### 测试
**测试方案**：全仓库不再出现 `http|opencode|proxy`、`EdgePipeline`。
**测试方法**：
- `grep -rn "http|opencode|proxy" framework/` → 0 处。
- `grep -rn "EdgePipeline" examples/ framework/` → 0 处。
**测试结果**：两处 grep 均 0 命中;`python -c "import framework"` 正常;355 passed。

---

## 问题 5：telemetry 定价表与实际模型脱节

### 问题
`DEFAULT_PRICING` 只有 gemini/gpt/claude,实际使用的免费模型 `hy3-free`(以及 sensenova-*)
落进 `default` 价($1/$3 per M),免费模型被估算成收费,成本报告失真。

### 方案
把 `hy3-free` 显式列为 $0/$0;未列模型仍落 default 价并加注释说明是占位估算。

### 修改
- `framework/utils/telemetry.py`:`DEFAULT_PRICING` 增 `"hy3-free": {0.0, 0.0}`,
  注释注明免费档与未列模型的回退语义。

### 测试
**测试方案**：`calculate_cost` 对 hy3-free 返回 0,对未知模型仍走 default 价。
**测试方法**：
- `python -c "from framework import calculate_cost; print(calculate_cost(1000, 500, 'hy3-free'))"` → 0.0。
- `calculate_cost(1000, 500, 'unknown-model')` → 按 default(记为占位)。
**测试结果**：hy3-free 成本 0、未知模型落 default;`pytest tests/test_memory_and_telemetry.py -q`
通过;355 passed。

---

## 问题 6：Executor 二次 run() 静默返回旧结果

### 问题
同一 `Executor` 实例跑完后再 `run()`:顶点已是终态,`_loop` 立即退出,返回状态快照且
`success=True`,不提示「未重跑」,调用方误以为重新执行了。

### 方案
加 `_has_run` 防护:同一实例第二次 `run()`/`stream()` 抛 `RuntimeError`,提示创建新
Executor(reset 后重建),不再静默返回旧结果。

### 修改
- `framework/executor/base.py`:`Executor.__init__` 加 `_has_run=False`;
  `stream()` 入口检查并置位。
- `tests/test_executor.py`:新增再跑会抛错的用例。

### 测试
**测试方案**：同一 executor 成功跑完后再次 `run()`,断言抛 `RuntimeError` 且首次结果正常。
**测试方法**：`pytest tests/test_executor.py::test_second_run_raises_instead_of_silent_stale -v`。
**测试结果**：首次 `success=True`;第二次 `RuntimeError`(match "already been run");
`test_executor.py` 全量通过;全量 355 passed。

---

## 结论
第二轮 6 个问题全部修复并回归锁定,`pytest tests/` = **355 passed**。