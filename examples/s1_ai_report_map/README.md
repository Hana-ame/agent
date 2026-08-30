# S1 AI Report (MapEdge)

> 按「问题 / 方案 / 修改 / 测试」记录：S1 论坛 AI 日报的 **MapEdge 版**实验，对比 opencode 直出。

## 问题 1：8 路手写扇出 vs 一条 MapEdge

### 问题
`s1_ai_report` 原版把「每帖 fetch+summarize」手写成 8 路扇出（`e_sel1-8`/`e_fetch1-8`/`e_sum1-8`），config 巨大、并发固定、不可复用。

### 方案
复用 `hn_ai_report` 的 MapEdge 模式：一条 `ProcessThreadsMap(MapEdge)` + `settings.pipeline`（fetch → summarize）对每个筛选出的帖子并发跑，`max_concurrency` 限流。

### 修改
- `examples/s1_ai_report_map/config.json`：`script: s1_edges.py:ProcessThreadsMap`，pipeline 两步。
- `examples/s1_ai_report_map/s1_edges.py`：`FetchEdge` / `SummarizeEdge` / `ProcessThreadsMap`。

### 测试
**测试方案**：MapEdge 对过滤后的每帖跑 pipeline 并汇聚。**测试方法**：
`python examples/s1_ai_report_map/demo.py`（proxy 内嵌）。**测试结果**：`report.md` 生成，多帖并发产出。

## 问题 2：24h 窗口拿不到旧楼主帖

### 问题
MapEdge 用「最近 24h 回帖」窗口，**旧楼主帖**（如 Qwen 配置教程帖）不在窗口内，MapEdge 版缺失、直出版（WebFetch 整页）有。

### 方案
记录为已知限制：窗口是「讨论动态」而非「历史归档」。直出路线补全历史内容。

### 修改
（无代码改动——数据窗口取舍，已记录于 `ai_report_notes.md` 与 `report.md` 对比表。）

### 测试
**测试方案**：同源 4 帖对比。**测试方法**：map + `opencode run --model opencode/hy3-free` 各跑一次，对比内容/成本。**测试结果**：map 报告含楼层号/具名用户可回溯；直出含旧帖全量；成本 map 为直出约 27%（见 `ai_report_notes.md` 问题 6 表格）。

## 文件

- `config.json`、`demo.py`、`s1_edges.py`、`vertex/report_hook.py`、`report.md`
- `opencode_direct.md`：同源直出对比报告。