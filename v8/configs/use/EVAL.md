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

## 定量结论（bootstrap，seed=42，20k 次）

### 唯一能确认的差异：sqlite_report 上 flash 更快

合并第一轮 + 第二轮 base 的 5 次/模型：

| | deepseek-flash | deepseek-v4-flash |
|---|---:|---:|
| p50 | **3.97s** | 15.09s |
| p95 | 7.26s | 45.60s |

- 中位延迟比（flash / v4）= **0.26**
- bootstrap 95% CI = **[0.08, 0.50]**
- 对应加速比 = **2.0×–12.5×**（点估计 **3.8×**）
- 质量：第二轮 base 两者都是 3.00/3；第一轮 flash 2.50、v4 3.00（n=2）
  → **没有稳定的质量差异，差异主要在延迟**

### 不能确认差异的：research / script

| DAG | p50 比（flash/v4） | 95% CI | 判断 |
|---|---:|---|---|
| research_brief | 0.33 | [0.06, 1.72] | CI 跨 1，不能确认 |
| script_judge | 0.82 | [0.31, 2.33] | CI 跨 1，不能确认 |

### workers=2：方向是“没有帮助”

research_brief：

| 模型 | w1 p50 | w2 p50 |
|---|---:|---:|
| deepseek-flash | 8.7s | 26.4s |
| deepseek-v4-flash | 26.3s | 83.9s |

n=2/组，只能当方向性证据；两个 http 只占 1.5s，瓶颈在后面的 LLM 合并。

### 方差

CV 0.32–1.04；research 单次 7.6–149.1s，sqlite 3.2–49.5s。
**单次运行不可用于比较；至少 5–10 次并看 P50/P95。**

### 结论边界

- 不是交错配对实验，网关漂移未完全控制
- 质量 rubric 是确定性的，不判断语义深度
- sqlite 的 3.8× 是当前窗口的观测值，不是跨 provider 的普遍常数

## 复现

```bash
cd v8
python scripts/sweep_use_dags.py
python scripts/analyze_use_sweep.py
```
