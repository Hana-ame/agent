# Examples

本目录包含 **19 个可运行实验**（另有 `scripts`、`s1profile_collect` 为辅助目录）。每个实验的
`README.md` 按「问题 / 方案 / 修改 / 测试」四段式记录：实验要解决的问题、方案、实际修改与实测结果。

> 所有示例统一入口：`python examples/run.py <示例>/config.json`（或该目录自带的 `demo.py/run.py`）。

## 概览索引

| 实验 | 解决的问题 | 运行方式 |
|---|---|---|
| `simple/` | 最简 3 节点串行管线怎么跑 | `python examples/run.py examples/simple/config.json` |
| `complex/` | 多源 fan-out/fan-in + 外部子类脚本 | `python examples/run.py examples/complex/config.json` |
| `conditional_routing/` | Guard 条件分发 + 级联剪枝不死锁 | `python examples/run.py examples/conditional_routing/config.json` |
| `custom_classes/` | 原生子类（Vertex/Edge）如何被识别加载 | `python examples/run.py examples/custom_classes/config.json` |
| `real_llm/` | 真实 LLM 端点 + 传输代理怎么配 | `python examples/run.py examples/real_llm/config.json` |
| `real_pi/` | 委派本地 `pi` CLI 子进程 | `python examples/run.py examples/real_pi/config.json` |
| `opencode_zen/` | 委派本地 `opencode` CLI | `python examples/opencode_zen/run.py` |
| `sensenova/` | 免代理直连 SenseNova 端点 | `python examples/run.py examples/sensenova/config.json` |
| `realtime_streaming/` | 非阻塞事件流观测 | `python examples/realtime_streaming/demo.py` |
| `self_correction/` | 业务重试 + 自纠错反馈 | `python examples/self_correction/demo.py` |
| `hitl_approval/` | HITL 暂停 + SQLite 快照恢复 | `python examples/hitl_approval/demo.py` |
| `subgraph/` | 嵌套子图 + 边界映射 | `python examples/subgraph/demo.py` |
| `simple_chain/` | 免 JSON 的程序化拓扑 | `python examples/simple_chain/demo.py` |
| `dynamic_topology/` | 运行时图增长 | `python examples/dynamic_topology/demo.py` |
| `race_mode/` | 先到先赢 + 取消滞后者 | `python examples/race_mode/demo.py` |
| `hn_ai_report/` | HN 端到端 AI 日报（MapEdge） | `python examples/hn_ai_report/demo.py` |
| `s1_ai_report/` | S1 直连版 AI 日报（8 路扇出） | `python examples/s1_ai_report/demo.py` |
| `s1_ai_report_map/` | S1 MapEdge 版 AI 日报 | `python examples/s1_ai_report_map/demo.py` |
| `finance_ai_report/` | 财经版 AI 日报（MapEdge 克隆，finance/politics 筛选） | `python examples/finance_ai_report/demo.py` |

辅助目录：`scripts/`（公共子类脚本）、`s1profile_collect/`（S1 数据收集，gitignored）。

---

## 问题：示例数曾与实际不符（16 → 实际 19）

### 问题
README 曾写「16 个示例」，漏掉 `s1_ai_report_map`、`sensenova`、`finance_ai_report` 三个已存在目录；且
`complex`/`custom_classes` 描述用「module hooks」旧措辞。

### 方案
数字改为实机统计；措辞改为「script 加载子类」。

### 修改
- 本索引：16→19，补 `s1_ai_report_map`/`sensenova`/`finance_ai_report` 三行；`complex`/`custom_classes` 条目改子类措辞。

### 测试
**测试方案**：索引行数=目录数。**测试方法**：`ls -d examples/*/ | wc -l`（21，排除 scripts/s1profile_collect
后为 19 个实验 + 2 辅助）与表格行数对比。**测试结果**：一致（19 行 = 19 目录）。

---

## 问题：`real_llm`/`real_pi` 曾由 `httpx` 直连与 `PiAgentRunner` 并存

### 问题
早期 `real_llm` 用裸 urllib 内联 HTTP 请求，绕开框架 agent；后 `real_pi` 通过注入
`PiAgentRunner` 以 `runner.py` 委派本地 CLI。两条路线并存令人困惑。

### 方案
统一为「script Edge 在 `__init__` 自持 agent」：`llm_edge.py:HttpLLMEdge`、`pi_edge.py:PiEdge`。

### 修改
- `real_llm/llm_edge.py`、`real_pi/pi_edge.py`：`__init__` 内 `self.agent = HttpLLMAgent/PiAgentRunner`；
  框架不再注入、无默认回退。
- examples/README 措辞同步。

### 测试
**测试方案**：两种 CLI agent（pi / opencode）都通过 script edge 工作。**测试方法**：
`grep -n \"self.agent\" examples/real_llm/llm_edge.py examples/real_pi/pi_edge.py examples/opencode_zen/zen_edge.py`。
**测试结果**：3 处均在 `__init__` 自持；测试 `tests/test_agents.py` 通过。

---

## 运行

```bash
# 通用 config 示例
python examples/run.py examples/simple/config.json
python examples/run.py examples/conditional_routing/config.json

# 自带 demo 的示例
python examples/realtime_streaming/demo.py
python examples/race_mode/demo.py
```

> 详细四段式说明见各示例目录的 `README.md`；实机报告见各目录 `report.md`。