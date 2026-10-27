# 单个 DSH session → 复盘报告 DAG

指定一个 DSH 历史 session，抽结论 → v7 生成 Markdown 复盘 → v8 落盘 → v9 存知识。

## 入口

```bash
cd /mnt/c/Users/lumin/Downloads/agent

# 1) 列出最近 session，挑一个
python3 v8/configs/dsh_session_report/run.py --list

# 2) 指定 session 跑报告（真实模型）
python3 v8/configs/dsh_session_report/run.py \
  --session session-adf80908-d20f-4ffa-b00f-952b9b4bf15d \
  --backend http

# 3) 网关不可用时，本地模板报告（不调模型，仍走完整 DAG）
python3 v8/configs/dsh_session_report/run.py \
  --session session-adf80908-d20f-4ffa-b00f-952b9b4bf15d \
  --backend local

# 4) 不用 run.py，直接跑 DAG（用 dag.json 里的默认 session）
cd v8
python3 -m dag8 run configs/dsh_session_report/dag.json --workers 1 --rpm 1
```

`--session` 支持完整 id 或唯一前缀。

## DAG

```text
collect.py     script  layer0  session_id → session JSON（title/request/conclusion/turns/tools）
report.json    v7 graph layer1  session JSON → Markdown 复盘
persist.py     script  layer2  report + session → 文件 + v9 claim
```

- `dag.json`：`seed_tag=session_id`，默认 session 写在 `seed_content`
- `run.py`：入口，负责传入目标 session、选后端、打印 report 路径
- `--backend http`：显式注入 v7 HTTP client（网关可用时最快）
- `--backend pi`：走 v8 默认 `PiLLMClient`（pi）
- `--backend local`：网关 502/超时时的本地降级，只用真实 session 数据拼模板，不调用模型

## 输出

| 文件 | 内容 |
|---|---|
| `reports/session_<slug>_report.md` | 最终复盘报告 |
| `reports/dsh_session_report_latest.md` | 最近一次报告 |
| `reports/session_<slug>.json` | 抽取的 session 数据（已脱敏） |
| `v9/data/dsh_session_report.sqlite3` | entity + claim，`source_url=dsh-session://<id>` |

## 已验证

用 `session-adf80908-d20f-4ffa-b00f-952b9b4bf15d`（“测试一下agent”，27 turns / 265 tools）跑通：

```text
collect   0.2s  3758 字符
summarize 0.0s  3660 字符   ← local backend
persist   0.2s   443 字符
report    3659 字符
```

真实模型后端本次被网关 502 挡住；`--backend local` 用于验证 DAG 全链路，网关恢复后去掉该参数即可。
