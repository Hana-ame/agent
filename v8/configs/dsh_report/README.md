# DSH 历史 session → 格式化报告

用 **v7（单 session 图 + merge 图）+ v8（并行调度 / 契约 / 轨迹）+ v9（知识库）** 把 DSH 历史会话结论做成 Markdown 报告。

## 数据源

- `~/.dsh/session-manager/state.json`：会话索引
- `~/.dsh/sessions/<cwd>/<session-id>/session.v4.jsonl.zstd`：会话记录（zstd JSONL）
- 只取当前工作区 `top-level` 会话，排除当前会话
- 结论 = 最后一个 assistant text block（跳过 reasoning）
- 抽取时自动脱敏 `sk-*` / `api_key` / `token` / `Bearer`

## 结构

```text
extract.py                 抽 6 个 session 结论 → reports/dsh_sessions_input.json
per_session.json           v7 单 session 图：结论 → 结构化 JSON
merge.json                 v7 merge 图：6 份结构化结论 → Markdown 报告
dag.json                   v8 总装：
                             g1..g6 (graph, 并行)
                             merge  (graph, join:any)
                             persist(script)
persist.py                 写报告/结构化 JSON + 写 v9 知识库
run_report.py              驱动：抽取 → dag8.run → 落盘
```

## 运行

```bash
cd /mnt/c/Users/lumin/Downloads/agent

# 完整 6 个 session
python3 v8/configs/dsh_report/run_report.py --workers 1 --rpm 1 --limit 6

# 快速验证 2 个
python3 v8/configs/dsh_report/run_report.py --workers 1 --rpm 1 --limit 2

# 只看抽取结果，不调模型
python3 v8/configs/dsh_report/extract.py --limit 6
```

## 输出

| 文件 | 内容 |
|---|---|
| `reports/dsh_sessions_report.md` | 最终 Markdown 报告 |
| `reports/dsh_sessions_structured.json` | 每个 session 的结构化结论 |
| `reports/dsh_sessions_input.json` | 抽取的原始结论（含脱敏） |
| `reports/dsh_report_run.json` | 本次 run 的状态、边、耗时 |
| `v9/data/dsh_report.sqlite3` | v9 知识库：entity + claim（source_url=`dsh-session://<id>`） |

## 运行策略

- **不缓存**：`use_cache=False`；每次重新抽取/生成
- **契约**：单 session 图必须输出 JSON（含 `session_id/topic/conclusion`）；merge 必须 Markdown（`#`）
- **无字数上限**：结论与报告不设长度
- **限流降级**：
  - 默认 `--workers 1 --rpm 1`，避免 429
  - `merge` 用 `join:"any"`，个别 session 失败也能出报告
  - `persist` 的 `o1..o6` 是 optional，等 `report` 才跑
  - v7 HTTP 客户端 `max_retries=5`，长时间退避重试
- **部分成功**：报告只陈述拿到的 session，不推断缺失项

## v9 里的知识

每个 session 结论写成：

```text
entity: 主题
claim : 主题 --session_conclusion--> <session-id>
        statement = 结论全文
        source_url = dsh-session://<session-id>
```

来源是本地会话记录，不是外部网页；`source_url` 用于可追溯，不代表外部证据。