# Finance AI Report (MapEdge) — 本地运行

> 按「问题 / 方案 / 修改 / 测试」记录：stage1st 财经版 AI 日报实验，`s1_ai_report_map` 的
> 财经话题克隆（fetch → LLM 筛选 finance/politics 话题 → ProcessThreadsMap 并发
> fetch-replies + summarize → `report.md`）。

## 问题 1：需要一个财经话题的端到端示例

### 问题
已有的 `s1_ai_report`/`s1_ai_report_map` 聚焦 AI 话题；财经/政策话题的抓取、筛选、
总结流程需要独立示例验证（不同站点版块、不同 URL 结构）。

### 方案
克隆 `s1_ai_report_map` 的 MapEdge 架构，筛选关键词换成财经/政治（finance/politics），
report 版块独立。

### 修改
- `examples/finance_ai_report/config.json`（已由并发会话落地，时间 04:25）。
- `examples/finance_ai_report/finance_edges.py`（MapEdge 管线 + fetch/summarize）。
- `examples/finance_ai_report/vertex/report_hook.py`（报告累积）。
- `examples/finance_ai_report/demo.py`。
- `tests/test_s1_edges.py`：追加 `finance_edges.py` 到 EDGE_PATHS（10 tests passed）。

### 测试
**测试方案**：财经帖子抓取 → LLM 筛选 → MapEdge 并发总结 → `report.md`。
**测试方法**：`python examples/finance_ai_report/demo.py`（proxy/端点内嵌 config）。
**测试结果**：`report.md` 生成（527 行，实机产出）；`pytest tests/test_s1_edges.py -q`
= **10 passed**。

## 已知限制（同 s1 map）
- MapEdge 24h 窗口拿不到旧楼主帖；直出路线才有历史全量。
- 筛选/总结 prompt 中文、不限条数（见 `s1_ai_report_map/README.md` 问题 2）。

## 文件

- `config.json`、`demo.py`、`finance_edges.py`、`vertex/report_hook.py`、`report.md`