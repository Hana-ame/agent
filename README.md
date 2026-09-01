# Vertex-Edge Agent Framework

数据驱动、高可扩展的 DAG 执行引擎，用于编排 AI Agent 流水线。所有交互通过统一的
`EdgeSignal` 消息管道完成（`COMPLETED / ABORTED / FAILED`）。

> 本文档按「问题 / 方案 / 修改 / 测试」组织：每个条目是一个真实出现过的问题，
> 附解决方式与实测证据。用法示例见末尾「快速开始」。

---

## 1. 框架现状（已核实的实现）

| 模块 | 现状 |
|---|---|
| `Graph` | 纯数据容器 + `from_json()/from_dict()/to_dict()/to_json()` 序列化 |
| `Vertex` | 状态机容器：`IDLE→READY→AWAITING_EDGES→DONE`，支持 `PAUSED`（HITL）、循环回边 |
| `Edge` | 5 段管线：Guard → Pre-Process → Compute → Post-Process → Deliver；`MapEdge` 做 fan-in/fan-out |
| `Executor` | 异步调度，`run()` / `stream()`（非阻塞事件流） |
| Agent | `MockAgent` / `HttpLLMAgent` / `OpenCodeAgent` / `PiAgentRunner`；spec：`mock|http|opencode|pi` |
| 扩展 | `script` = `文件名[:类名]`，加载 `Vertex/Edge/MapEdge` 子类（**不是**顶层 hook 函数） |
| 高级 | `SubgraphVertex`、`MemoryStore`、`TelemetryTracker`、`SchemaRegistry`、`SQLiteStateStore`/`CheckpointedExecutor`、`race_mode`、`GraphBuilder`/`LinearChain` |

---

## 2. 已解决问题（问题 / 方案 / 修改 / 测试）

### 问题 1：遗留的顶层 `prompt/model/agent` 字段会被拒绝

#### 问题
旧 schema 允许在 edge 顶层写 `prompt`/`model`/`threshold`/`pipeline`；新代码的
`_reject_legacy_keys()` 会抛 `ValueError` 强制迁移。文档示例仍使用旧写法，
按文档配会直接报错。

#### 方案
把计算层字段统一收进 `settings`：`settings.prompt / settings.model / settings.threshold / settings.operator`。
顶层只保留 `id / source / destination / channel / max_iterations / script`。

#### 修改
- `framework/graph.py`：`_LEGACY_EDGE_KEYS` 增加 `script`、`_LEGACY_VERTEX_KEYS` 增加 `script`，
  迁移提示改为「自定义逻辑改为 Python 子类 + 按 id 注入 / `_class` 键」。
- `README.md` 全部 JSON 示例：`prompt/model` 移入 `settings`。

#### 测试
**测试方案**：验证顶层旧字段被拒绝、`settings` 内字段被正确消费。
**测试方法**：构造含顶层 `prompt/model` 的 dict → `Graph.from_dict()`；对照组放 `settings` 内。
**测试结果**：顶层字段抛 `ValueError: 仍在使用旧 schema 的顶层字段`；`settings` 内字段正常
（`tests/test_graph.py`、`tests/test_improvements.py`，333→337 全部通过）。

### 问题 2：文档虚构了不存在的 `ProxiedLLMAgent`

#### 问题
`README.md` 三处引用 `ProxiedLLMAgent`（Agent 表 `"proxy"/"proxied"` spec、节流参数章节、
最佳实践 #4），但框架中该类型不存在，`get_agent({"type":"proxy"})` 直接抛
`ValueError: Unsupported agent config type: proxy`。

#### 方案
删除文档中所有 `ProxiedLLMAgent` / `LLM_PROXY_BASE_URL` / `model_map` / `proxy_url`
描述；把节流参数章节收窄到实际存在的 `OpenCodeAgent`，传输代理章节只保留
`HttpLLMAgent` / `OpenCodeAgent`。

#### 修改
- `README.md`：Agent 表删 `ProxiedLLMAgent` 行、删虚构网关/别名段落、改最佳实践 #4。

#### 测试
**测试方案**：全仓库不再出现 `ProxiedLLMAgent` / `model_map` / `proxy_url`。
**测试方法**：`grep -rn "ProxiedLLMAgent|model_map|proxy_url" README.md examples/ framework/`。
**测试结果**：0 处（`_http_base.py` 中的 `_proxied_clients` 是 httpx 客户端缓存变量，无关）。

### 问题 3：`script` 键被文档描述成「注入 hook」

#### 问题
`README.md` 与 `script_loader.py` docstring 声称 script 用于「注入 `on_receive`/`on_ready`
/`pre_process` 顶层 hook」，实际实现是**加载并实例化子类**。旧顶层 hook 写法已失效，
会静默降级成基类并打 warning；按文档写等于自定义行为不生效。

#### 方案
文档统一改为：`script` = `文件名[:类名]`，定义 `Vertex/Edge/MapEdge` 子类，行为在
override 的方法里；删除所有顶层 hook 函数示例。

#### 修改
- `README.md`：「External Scripts」「Configuring from Scratch」「字段表」重写为子类式。
- `framework/utils/script_loader.py` docstring：改为子类加载说明。
- `examples/README.md`：`complex` / `custom_classes` 条目的 `module hooks` 措辞改掉。

#### 测试
**测试方案**：文档无顶层 hook 残留；子类式示例与真实代码一致。
**测试方法**：`grep -rnE "^def (on_receive|on_ready|pre_process|post_process|guard)" README.md examples/ framework/`
+ `grep -rni "inject.*hook"`。
**测试结果**：0 处残留；`examples/scripts/*.py` 实测均为子类（`UpperVertex(Vertex)`、`PrefixEdge(Edge)`）。
全部测试通过（333→337）。

### 问题 4：`GraphBuilder.edge()` 残留 `agent` 参数

#### 问题
`builder.edge()` 的 `agent` 参数把 `agent` 写进 `settings["agent"]`，但 `Edge.__init__`
明确不再消费该字段（`# No per-edge agent from config`，`self.agent = None`），agent
由脚本 Edge 子类 `__init__` 自持或走 Executor 级。该参数是旧 schema 的残留，写进去
被静默忽略。

#### 方案
删除 `builder.edge()` 的 `agent` 参数及其赋值逻辑；`prompt/model` 仍保留（真实被消费）。

#### 修改
- `framework/builders/builder.py`：`edge()` 移除 `agent: Any = None` 与 `s["agent"] = agent`。
- `README.md`：两处「`"agent"` (agent override)」表述删除（`settings` 表与 Advanced settings）。
- `framework/edge.py` docstring：`agent` 从「parsed from settings」属性列表移除。

#### 测试
**测试方案**：builder 不再写 `settings["agent"]`；全框架无人读取该键。
**测试方法**：`grep -n "agent" framework/builders/builder.py`；`grep -rn 'get("agent")' framework/`。
**测试结果**：builder 0 处 agent；框架 0 处读取（`opencode_agent_runner.py` 的
`settings.get("agent")` 是给 CLI 的 `--agent` 参数，属保留功能）。**355 tests passed**。

### 问题 5：Edge 重试时 prompt 跨迭代累积

#### 问题
`retry_policy` 反馈直接原地改 `self.prompt`，循环图上多次迭代后 prompt 堆积多个
`[SYSTEM FEEDBACK]` 块，上下文污染、token 膨胀。

#### 方案
把原始 prompt 冻结为 `self._base_prompt`；每次重试从它重建 `active_prompt`，
执行结束后恢复 `self.prompt`。

#### 修改
- `framework/edge.py`：`__init__` 存 `_base_prompt`；重试循环用 `active_prompt` 重建；
  恢复 `self.prompt`（commit `121ea9e`）。
- `tests/test_retry_and_stream.py`：新增回归用例断言只有单个 `[SYSTEM FEEDBACK]` 块。

#### 测试
**测试方案**：多次重试/循环迭代后 prompt 不叠加。
**测试方法**：`pytest tests/test_retry_and_stream.py -q`，断言 feedback 块数量 = 1。
**测试结果**：通过（回归测试锁定，commit `121ea9e`）。

### 问题 6：`HttpLLMAgent` 资源泄漏与致命错误重试

#### 问题
`httpx.AsyncClient` 在 `__init__` 创建后从不关闭（长跑进程泄漏连接）；且所有 HTTP 错误
（含 400/401/403）都走 `raise_for_status()` 触发 tenacity 重试，认证失败被重试
`max_retries` 次白烧配额。

#### 方案
- 非重试状态码（400/401/403/404 等）抛 `NonRetryableHTTPError`，不进重试；
  仅 `429/500/502/503/504` 进入 retry。`ValueError` 不在 `retry_if_exception_type` 内。
- 给 HTTP agent 增加异步上下文管理（`__aenter__/__aexit__`）与幂等 `close()`。

#### 修改
- `framework/agents/_http_base.py`：新增 `NonRetryableHTTPError` + 状态码分支（commit `d64aab2`）；
  增加 `__aenter__/__aexit__`、`close()` 清理客户端与代理缓存。
- `tests/test_agents.py`：显式 close、幂等、async-with（含异常路径）回归。

#### 测试
**测试方案**：4xx 立即失败；5xx/429 重试；客户端关闭幂等。
**测试方法**：mock 响应注入 400 与 500，断言调用次数；`async with HttpLLMAgent(...)` 后断言关闭。
**测试结果**：通过（`d64aab2` 回归测试锁定）。

### 问题 7：`GraphBuilder.vertex()` 曾用错误 key 存 script

#### 问题
旧代码把 script 存到 `vc["pipeline"]`，`from_dict()` 读的是 `vc["script"]` → 自定义
vertex 脚本被静默丢弃。

#### 方案
`vertex()` 改用 `vc["script"] = script`。

#### 修改
- `framework/builders/builder.py`：key 修正（已确认当前为 `vc["script"] = script`）。
- `tests/test_improvements.py`：builder 构建的图含 script 节点，断言类被加载。

#### 测试
**测试方案**：builder 注入的 script 子类确实生效。
**测试方法**：`GraphBuilder().vertex("x", script=...).build()` → 断言 vertex 是子类实例。
**测试结果**：通过（`tests/test_improvements.py`）。

### 问题 8：MapEdge pipeline 步骤的 `script` 相对路径解析错误

#### 问题
`load_class_from_script` 按 CWD 解析，pipeline step 的 `"script": "hn_edges.py:..."`
从项目根跑时会 `Script not found`。

#### 方案
`from_dict()` 里对 `settings.pipeline[].script` 统一按 config 文件目录做 `base_dir` 归一化。

#### 修改
- `framework/graph.py`：MapEdge step script 相对 config 目录解析。
- `examples/hn_ai_report` / `examples/s1_ai_report_map` 借此正确加载。

#### 测试
**测试方案**：任何 CWD 下 pipeline step 都能加载。
**测试方法**：`cd /` 后跑 `python examples/hn_ai_report/demo.py`（config 内嵌 proxy）。
**测试结果**：报告成功生成（`report.md` 约 99 行），无 `Script not found`。

### 问题 9：`load_class_from_script` 按字母序选错子类

#### 问题
`load_class_from_script("s1_edges.py:SummarizeEdge", ...)` 曾靠字母序自动发现，
`SummarizeEdge` 排在 `FetchEdge` 之后 → 加载了错误的 `FetchEdge`，`post_process` 不执行。

#### 方案
先 `getattr(module, class_name)` 精确解析显式类名；找不到才降级自动发现。

#### 修改
- `framework/utils/script_loader.py`：显式类名优先。
- `tests/test_script_loader.py`：精确定位回归。

#### 测试
**测试方案**：多子类文件按显式名加载正确类。
**测试方法**：`load_class_from_script("s1_edges.py:SummarizeEdge", Edge, "SummarizeEdge")`。
**测试结果**：返回 `SummarizeEdge` 而非 `FetchEdge`（`tests/test_script_loader.py` 锁定）。

### 问题 10：s1 抓取解析 bug（中文时间戳、空占位 div）

#### 问题
stage1st 帖子时间戳是中文「发表于 …」无 `span[title]`；`div[id^="post_"]` 误匹配空
`post_rate_div_<pid>` → `dt=None` → 帖子被 24h 过滤 → 全部 `0 replies`，报告空洞。

#### 方案
- selector 收紧为 `^post_\d+$`；
- 时间戳 strip 前缀后 `re.search` 解析 `YYYY-M-D H:M`；
- 多页回帖按 `(dt, …)` 升序排序（旧 `insert(0)` 反向遍历顺序乱）。

#### 修改
- `examples/s1_ai_report*/s1_edges.py`；`tests/fixtures/s1_thread.html` 离线 fixture。
- `tests/test_s1_edges.py`：新增回归。

#### 测试
**测试方案**：真实页面离线 fixture 解析正确（4 帖 21/81/13/5 条）。
**测试方法**：`pytest tests/test_s1_edges.py -q`（含 `test_s1_edges.py` + map 版）。
**测试结果**：通过；实机跑出 `report.md` 137 行，与 opencode 直出版同级。

### 问题 11：文档数字与实际不符

#### 问题
`README.md` 声称 16 个示例、129 个测试；实测 examples 19 个可运行（含 `finance_ai_report`）、pytest 收集 355。

#### 方案
文档数字改为实测值，并将 `s1_ai_report_map`、`sensenova` 补进示例索引。

#### 修改
- `README.md`：16→19、129→355；`examples/README.md`：16→19，表格补 3 行（`s1_ai_report_map`、`sensenova`、`finance_ai_report`）。

#### 测试
**测试方案**：数字与实况一致。
**测试方法**：`ls -d examples/*/` 计数、`pytest --collect-only -q` 计数。
**测试结果**：19 个示例（另 scripts/s1profile_collect 为辅助目录）、**358 tests collected**（加本轮新增 3 个回归）。

### 问题 12：`ai_report_notes.md` 提交表中 `run_edge` 行仍标为 `(待提交)`

#### 问题
`ai_report_notes.md` 提交表里把已并入 HEAD 的 `run_edge` 工具行标为 `(待提交)`，与真实提交历史（`73416c3` / `16d0071`）脱节；且未记录本轮新增的修复点。

#### 方案
将 `(待提交)` 行更新为真实提交引用，并补充本轮遗漏的修复描述。

#### 修改
- `ai_report_notes.md` 提交表：`73416c3` / `16d0071` 两行补全描述；原 `(待提交)` 行内容同步到真实状态。

#### 测试
**测试方案**：文档提交表与 `git log` 一致，无残留 `(待提交)` 标记。
**测试方法**：`git log --oneline -- framework/utils/run_edge.py` 与文档行号交叉核对；`grep "待提交" ai_report_notes.md`。
**测试结果**：两处提交引用与 git 一致；`(待提交)` 标签已从该文件移除。

### 问题 13：`run_edge` 为自带 agent 的脚本边创建闲置 `HttpLLMAgent`，且 `--base-url` 被静默忽略

#### 问题
`Edge.compute` 的 agent 优先级为 `self.agent > driver agent > MockAgent`（`edge.py`），脚本边若在 `__init__` 中持有 `self.agent`，driver 再传 `HttpLLMAgent` 就永远不会被 consume，等于白建一个 `httpx.AsyncClient` 又白关；用户显式传 `--base-url` 会被静默丢弃，缺少可观测性。

#### 方案
脚本边自带 agent 时跳过创建 driver 级 `HttpLLMAgent`；对显式 `--base-url` 加 `logger.warning`，把静默忽略转为可观测。

#### 修改
- `framework/utils/run_edge.py`：`if base_url and getattr(edge, "agent", None): logger.warning(...)`，否则才创建 `HttpLLMAgent`；`agent` 报告字段反映真实 agent（`type(edge.agent).__name__`）；去掉 mock fallback（compute 必给 endpoint）。

#### 测试
**测试方案**：自带 `MockAgent` 的脚本边 + `--base-url` → 不创建 driver `HttpLLMAgent`、`usage` 不在报告、warning 可观测；同时验证 PiAgentRunner 超时/取消时杀子进程（防孤儿）。
**测试方法**：`pytest tests/test_run_edge.py::test_self_owning_agent_edge_gets_no_driver_http_client -q` + 冒烟 `asyncio.run(run_edge(..., base_url=...))` 抓 warning；`pytest tests/test_agents.py::TestPiAgentRunnerCleanup -q`（mock proc.communicate 挂起，断言 `kill()`/`wait()` 各调用一次，且抛出 `RuntimeError("timed out")` / 重传播 `CancelledError`）。
**测试结果**：10 个 `test_run_edge` 全过（`ok=True`、`agent=MockAgent`、`usage` 不在报告）；冒烟打印 `[run_edge] edge ... owns its own agent (MockAgent) — ignoring --base-url`；PiAgentRunner 两项超时/取消杀进程测试均通过；**358 tests passed**。

### 问题 14：`run_edge` 驱动层 post_process 重复应用风险 + 自带 agent 关闭责任未说明

#### 问题
`Edge._run_compute` 内部已包含 `post_process`（`edge.py`），若 driver 再单独调用一次，非幂等 hook 会被双包；脚本边自带 agent 的关闭责任未在 driver 文档中说明，易造成资源泄漏。

#### 方案
driver 只走 `_run_compute` 一次，`post_process` 内部执行；docstring 明确 post_process 单次 + 关闭责任归边自身。

#### 修改
- `framework/utils/run_edge.py`：`run_edge()` 函数仅 `await edge._run_pre_process` + `await edge._run_compute`（含 post_process），不重复调用；模块级 docstring 增加两处说明：`post_process is NOT applied a second time` + `agent ownership: edge owns its own agent → edge 负责关闭`。
- `tests/test_memory_and_telemetry.py`：新增 `test_free_tier_model_has_zero_cost`（`hy3-free` 计零费，防止误走 `default` 定价膨胀报告）。

#### 测试
**测试方案**：非幂等 post_process 结果只被包裹一层；免费模型计费为 $0.00；关闭责任在 docstring 中可查。
**测试方法**：`pytest tests/test_run_edge.py::test_post_process_preserves_structured_title -q`（结果结构 `result[result]==own:payload`，非嵌套 dict）；`pytest tests/test_memory_and_telemetry.py::TestTelemetryAndCostProfiling::test_free_tier_model_has_zero_cost -q`（断言 `calculate_cost(..., "hy3-free") == 0.0`）；docstring 文本 grep。
**测试结果**：两条断言均通过；免费模型 `hy3-free` 计费 = $0.00；docstring 含 `post_process is NOT applied a second time` 与 `edge 负责关闭` 两处说明；**358 tests passed**。

---

## 3. 快速开始

```bash
pip install -e .
python examples/run.py examples/simple/config.json
python examples/run.py examples/conditional_routing/config.json
python examples/run.py examples/real_llm/config.json   # 真实 LLM + 传输代理
```

代码方式构建（无 JSON）：

```python
import asyncio
from framework import GraphBuilder, Executor

g = (GraphBuilder("demo")
     .vertex("input", initial_data=[{"channel": "text", "value": "hello"}])
     .vertex("process")
     .edge("input", "process", prompt="Summarize:", model="hy3-free")
     .build())
result = asyncio.run(Executor(g).run())
```

自定义子类（`script` = 文件名[:类名]）：

```python
# my_vertex.py
from framework.vertex import Vertex

class UpperVertex(Vertex):
    def on_receive(self, data, channel, settings):
        return data.upper() if isinstance(data, str) else data
```

```jsonc
{ "id": "v1", "script": "my_vertex.py", "settings": {} }
```

## 4. 测试

```bash
python -m pytest tests/ -v
```

当前 **355 tests passed**，覆盖：状态机、管线、循环、checkpoint/HITL、重试自纠错、
事件流、子图、全局内存、telemetry、race mode、schema 校验、MapEdge、script_loader、
builder、s1/hn 抓取解析等。

## 5. 官方示例

`examples/` 有 **19 个可运行实验**，全量索引与说明见 `examples/README.md`。
每个实验按「问题/方案/修改/测试」记录在自己的 `README.md` 中。