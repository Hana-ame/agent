# S1 AI Report — 直出 8 路扇出版

> 按「问题 / 方案 / 修改 / 测试」记录：S1 论坛 AI 日报的**手写 8 路扇出**实验版，是
> `s1_ai_report_map`（MapEdge 版）的对比基线。

## 问题 1：手写扇出能跑但不可扩展

### 问题
`config.json` 用 `e_sel1-8`/`e_fetch1-8`/`e_sum1-8` 手写 8 路并行（每帖一条 fetch+summarize），
配置约 27 处 script 引用，帖子数一变就要改图。

### 方案
保留为「直出对比基线」：证明手写扇出**可正常产出报告**，同时作为 MapEdge 版的对照
（验证 MapEdge 产物同质、成本更低）。

### 修改
- `examples/s1_ai_report/config.json`：8 路扇出 + `v_report`（`vertex/report_hook.py`）。
- `examples/s1_ai_report/s1_edges.py`：`FetchThreadsEdge`/`FilterEdge`/`SelectEdge`/`FetchEdge`/`SummarizeEdge`。
- `examples/s1_ai_report/demo.py`（已核实存在）。

### 测试
**测试方案**：8 帖并发 fetch+summarize 产出 `report.md`。**测试方法**：
`python examples/s1_ai_report/demo.py`（proxy 内嵌）。**测试结果**：`report.md` 生成；
与 `s1_ai_report_map` 对比——内容同质、手写版配置更重、MapEdge 版成本约 27%（见
`examples/s1_ai_report_map/README.md` 问题 2 与 `ai_report_notes.md` 问题 6 表格）。

## 文件

- `config.json`、`demo.py`、`s1_edges.py`、`vertex/report_hook.py`、`report.md`