# HN AI Report — 端到端 AI 日报（MapEdge）

> 按「问题 / 方案 / 修改 / 测试」记录：Hacker News AI 日报实验，MapEdge 架构的
> 端到端示范（fetch → 筛选 → 每帖并发 fetch+summarize → 聚合成 `report.md`）。

## 问题 1：12 顶点 + 17 边手写 fan-out 太重

### 问题
早期 HN 示例手写每条路径（每帖独立 fetch/summarize 边），config 120+ 行、并发固定、
不可扩展。且 `hn_edges.py` 与 config 内联代码曾经双轨并存、令人困惑。

### 方案
走「类扩展 + MapEdge」路线：
- 自定义逻辑全部在 `hn_edges.py` 子类（`FetchTopStoriesEdge`/`FilterEdge`/
  `FetchCommentsEdge`/`SummarizeEdge`/`ProcessStoriesMap`）；
- config 用 `script: hn_edges.py:ClassName` 显式引用，删除内联代码；
- 每帖处理收敛为一条 `ProcessStoriesMap(MapEdge)` + `settings.pipeline`。

### 修改
- `examples/hn_ai_report/config.json`：约 40 行；`script` 显式类名；proxy 内嵌 settings。
- `examples/hn_ai_report/hn_edges.py`（已核实存在）。
- `examples/hn_ai_report/demo.py`（已核实存在）。

### 测试
**测试方案**：端到端产出 HN AI 日报。**测试方法**：
`env -u HTTPS_PROXY -u HTTP_PROXY python examples/hn_ai_report/demo.py`（proxy 在 config 内）。
**测试结果**：`report.md` 生成（约 99 行）；筛选不限制条数；报告中文、含 `# [标题](链接)`。

## 问题 2：MapEdge pipeline 的 script 相对路径与显式类名

### 问题
pipeline step 的 `script` 曾按 CWD 解析（→ Script not found）；自动发现曾按字母序
选错子类（`SummarizeEdge` 被 `FetchEdge` 抢）。

### 方案
- step script 按 config 目录归一化（`framework/graph.py`）；
- `load_class_from_script` 显式类名优先（`framework/utils/script_loader.py`）。

### 修改
- `framework/graph.py`、`framework/utils/script_loader.py`（均已核实修复）。

### 测试
**测试方案**：任意 CWD 下 pipeline step 加载正确类。**测试方法**：
`pytest tests/test_script_loader.py -q` + 实机 demo。**测试结果**：通过；
报告含逐帖总结（`SummarizeEdge.post_process` 生效）。

## 问题 3：token/耗时对比（map vs opencode 直出）

### 问题
需要量化 MapEdge（框架 pipeline）与 `opencode run` 直出两条路线的成本差异。

### 方案
同源 HN 帖两条路线各跑一次，读真实 usage（框架 `get_usage_summary()` + opencode SQLite）。

### 修改
- `framework/agents/_http_base.py`：真实 token 捕获（`usage_log`/`get_usage_summary`）。实机对比见 `ai_report_notes.md` 问题 6。

### 测试
**测试方案**：两条路线 token/耗时对比。**测试方法**：分别跑 demo 与
`opencode run`，读数据。**测试结果**：map 总消耗约直出 **32%**（HN），input 6-9k vs
78k（直出 WebFetch 整页）；见 `ai_report_notes.md` 表格。

## 文件

- `config.json`、`demo.py`、`hn_edges.py`、`vertex/report_hook.py`、`report.md`、`opencode_direct.md`