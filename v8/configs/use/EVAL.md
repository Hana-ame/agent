# 三个 DAG 的参数搜索（真实运行）

> 只统计真实网关运行；质量用确定性 rubric（0–4 / 0–3），不调额外评审模型。

## 实验

- DAG：`research_brief` / `sqlite_report` / `script_judge`
- 模型：`deepseek-flash` / `deepseek-v4-flash`
- 重复：第一轮每个组合 2 次；`research_brief` 额外比较 workers=1/2
- 第二轮：`sqlite_report` 加“必须提最高+最低”的 strict prompt，每组合 3 次
- 原始数据：`v8/data/use_sweep.json`、`v8/data/use_sweep_sqlite.json`

## 结果

| DAG | 模型 | 质量 | p50 | 成功 |
|---|---|---:|---:|---:|
| research_brief | deepseek-flash | 4.00 | **8.7s** | 4/4 |
| research_brief | deepseek-v4-flash | 4.00 | 26.3s | 4/4 |
| sqlite_report（第一轮） | deepseek-flash | 2.50 | **3.7s** | 2/2 |
| sqlite_report（第一轮） | deepseek-v4-flash | 3.00 | 30.6s | 2/2 |
| sqlite_report（第二轮 base） | deepseek-flash | 3.00 | **4.1s** | 3/3 |
| sqlite_report（第二轮 base） | deepseek-v4-flash | 3.00 | 10.3s | 3/3 |
| sqlite_report（strict） | deepseek-flash | 3.00 | 7.4s | 3/3 |
| sqlite_report（strict） | deepseek-v4-flash | 3.00 | 14.1s | 3/3 |
| script_judge | deepseek-flash | 3.00 | **33.3s** | 2/2 |
| script_judge | deepseek-v4-flash | 3.00 | 40.6s | 2/2 |

## 结论

1. **这三个 DAG 上，deepseek-flash 都比 v4-flash 快**；v4-flash 没有体现出更快。
2. **workers=2 对 research_brief 没帮助**：两个 http 各 1.5s，真正耗时是后面的 LLM 合并；加并发只是多等模型。
3. **sqlite 的 base prompt 已经够用**；strict prompt 质量一样，延迟更高。若不能接受“漏掉最低区域”，再加 strict。
4. **script_judge 质量稳定**，judge 闸门没有拒绝；耗时主要来自改写模型。
5. **方差极大**：research 单次 7.6–149.1s，sqlite 3.2–49.5s；2–3 次不足以稳定排名。

## 推荐

- 默认：`deepseek-flash` + `workers=1` + 现有 prompt
- `research_brief` 保持 2 路 http 并行，但不要再加 workers
- `sqlite_report` 先 base prompt；出现漏项再加 strict
- 想要稳定排名，至少 5–10 次重复并看 P50/P95，而不是单次

## 复现

```bash
cd v8
python scripts/sweep_use_dags.py
```
