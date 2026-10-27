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
collect.py          script  layer0  session_id → session JSON（title/request/conclusion/turns/tools）
a_background        v7 graph layer1  背景与用户诉求
a_process           v7 graph layer1  过程与关键动作
a_decisions         v7 graph layer1  技术决策与判断
a_conclusion        v7 graph layer1  最终结论与证据
a_risks             v7 graph layer1  风险与未决
a_reuse             v7 graph layer1  可复用经验与下一步
merge               v7 graph layer2  6 个维度整合成一份 Markdown
persist.py          script  layer3   report + session → 文件 + v9 claim
```

6 条 aspect edge 同层并行；merge 用 `join:"any"` + optional，个别维度失败也能出报告。

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
collect       0.4s  3758 字符
a_background  0.0s   116 字符
a_process     0.0s    50 字符
a_decisions   0.0s    40 字符
a_conclusion  0.0s  3346 字符
a_risks       0.0s    32 字符
a_reuse       0.0s    36 字符
merge         0.0s  3719 字符
persist       0.1s   443 字符
report        3718 字符
```

`--backend local` 关闭了限流（本地不需要 rpm），用于验证 DAG 全链路；真实模型后端本次被 moonchan 网关 502 挡住，恢复后把 `local` 换成 `http` 即可。
