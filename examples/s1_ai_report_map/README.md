# S1 AI Report (MapEdge)

`s1_ai_report` 的 MapEdge 版克隆 —— 参考 `examples/hn_ai_report` 的
`ProcessStoriesMap` 模式，把原版 s1 的显式 8 路扇出（e_sel1-8 / e_t1-8 / e_sum1-8）
收敛成**一条 MapEdge + 并发 pipeline**。

## 图结构

```
v_forum ──FetchThreadsEdge──▶ v_threads ──FilterEdge(llm hy3-free)──▶ v_router
    ──ProcessThreadsMap(MapEdge)──▶ v_report(ReportVertex)
```

MapEdge 的 `settings.pipeline` 定义每帖的加工管线：

```json
"pipeline": [
  { "type": "fetch", "script": "s1_edges.py:FetchEdge" },   // 抓该帖回复(最近24h)
  { "type": "llm",   "script": "s1_edges.py:SummarizeEdge", // hy3-free 逐帖总结
    "prompt": "...", "model": "hy3-free" }
],
"max_concurrency": 5
```

列表里每个被过滤出的帖子在 `asyncio.gather` 下并发走这条管线，结果逐条
`receive_signal` 给 `v_report`，由 `ReportVertex` 累加写 `report.md`。

## 本地实机跑

```bash
cd 仓库根目录
HTTPS_PROXY=http://127.0.1.6:7890 python examples/s1_ai_report_map/demo.py
# 免费：hy3-free (opencode.ai/zen/v1) + Clash 免费代理池
```

报告写到 `examples/s1_ai_report_map/report.md`。

## MapEdge vs opencode prompt 直出（同源 4 帖对比）

同一批 S1 帖子，两条路各跑一次（都走免费模型）：

| 维度 | MapEdge（本示例） | opencode prompt 直出（`opencode run --model opencode/hy3-free`） |
|---|---|---|
| 架构 | 静态并发管线：每帖 fetch+总结，确定性、可审计 | 单 agent 一次 prompt，自主决定抓取与综合（黑盒 agentic 循环） |
| 抓取深度 | `FetchEdge` 只取**最近 24h 回复** → 多数帖子 0 回复，总结空泛 | opencode **WebFetch 整页** → 拿到 297 页专楼、Time100 榜单细节等完整内容 |
| 输出 | 每帖独立小结（AI 趋势/用户观点/关键论点），信息较薄 | 单篇连贯报告，趋势/观点/论点详实、可引用原文 |
| 成本/并发 | 每帖一个 LLM 调用，可水平扩展，上限可预估 | 一个 agent 循环，灵活但耗时/调用数不确定 |
| 可复现 | 每步输入输出确定，易于调试与审计 | 结果依赖 agent 自主行为，不易复现 |

**结论**：`report.md`（MapEdge）与 `opencode_direct.md`（直出）对比可见，
直出信息密度显著更高。差距主要来自两点：(1) MapEdge 的 `FetchEdge` 限了
`hours=24` 过滤，老帖最近无回复 → 抓取内容少；(2) opencode 是 agentic，
会自行决定抓整页并把多帖信息交叉综合。MapEdge 的优势是**确定性 + 可审计 +
成本可控**，适合"每项输入有限、可并行"的稳定场景；信息深度优先时直出更胜一筹。

## 文件

- `config.json` —— 图定义（script:文件:类 + settings.prompt/model，无 agent 字段）
- `s1_edges.py` —— FetchThreadsEdge / FilterEdge / ProcessThreadsMap(MapEdge) + 管线步骤
- `vertex/report_hook.py` —— ReportVertex，累加写 report.md
- `demo.py` —— Executor(graph) + 顶层 HttpLLMAgent（MapEdge 管线步骤需要）
- `report.md` —— MapEdge 实机跑出的报告
- `opencode_direct.md` —— `opencode run --model opencode/hy3-free` 同源直出报告
