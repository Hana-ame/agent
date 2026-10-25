# v7 / v8 / v9 使用指南与自评指标

## 1. 先选层

```text
一张图内完成任务            → v7
多张图要并行/合并/契约       → v8
要长期积累事实与关系         → v9
```

| 层 | 负责 | 不负责 |
|---|---|---|
| v7 | 标签匹配、边执行、单图 step | 多图组织、长期知识 |
| v8 | 波次并行、合并、契约、pi/script/http/db、SQLite 轨迹 | 跨 run 知识 |
| v9 | 实体/关系/来源/推理循环 | 通用调度（复用 v8） |

## 2. v7：DAG 本体

```python
import v7
v7.load("graph.json")
v7.step()
print(v7.contents)
v7.set("external", "外部结果")
v7.step()
```

**最佳实践**

- 一张图只做一类转换；tag 名稳定、可预测
- 需要外部处理时，`step()` 到卡点 → 外面算完 `set()` → 继续
- 不把无关历史塞进 prompt；每条边只取需要的 tag
- 匹配靠精确 tag；不要用近似名（`report` vs `error_report`）

**指标**

| 指标 | 定义 | 看法 |
|---|---|---|
| `steps_to_result` | 到 result 的 step 数 | 越少越好 |
| `success_rate` | 有 result 的运行 / 总运行 | 1.0 |
| `latency_p50` | 到 result 的中位耗时 | 越低越好 |
| `context_chars` | 每条边 prompt 字符数 | 越小越干净 |

## 3. v8：组织层

```bash
python -m dag8 run configs/v7_core/dag_parallel.json --workers 3 --rpm 3
python -m dag8 run configs/use/research_brief.json --workers 2 --rpm 3 --json
```

**最佳实践**

- 只在有**独立分支**时上 v8；单链任务用 v7
- 分支用 `when`；汇聚用 `join:"any"`，避免剪枝后死等
- 输出加契约：`expect: {min_chars, max_chars, type:"json"}`；限制结果长度
- 确定性步骤用 `script`；语义步骤用 `pi`；外部数据用 `http/db`
- 当前策略：**不用缓存**；每次重新算，避免旧结果污染
- 慢调用用 `--rpm`/`--max-seconds` 控，不靠加 workers 硬顶

**指标**

| 指标 | 定义 | 当前实测 |
|---|---|---|
| `parallel_speedup` | 串行墙钟 / 并行墙钟 | **2.7×**（3 个短 v7 子图：10.83s → 4.0s） |
| `contract_pass_rate` | 契约通过边 / 执行边 | 3/3（research/sqlite/script） |
| `failure_rate` | error 边 / 执行边 | 0 |
| `output_chars` | 结果字符数 | 108 / 29 / 19（受 max_chars 限制） |
| `model_p50_ratio` | 快模型 p50 / 慢模型 p50 | 0.26（sqlite：3.97s vs 15.09s，95% CI 0.08–0.50） |

## 4. v9：知识推理

```bash
python -m v9.demo                # 完整 demo：推理 + 实体扩展 + 多跳
```

一个推理周期也可以是一张 v8 DAG（`v9/configs/cycle.json`）：
两路 http 并行 → v7 图生成 → script 校验入库。

**最佳实践**

- claim 必须有 `source_url` 或 `derived_from`；无来源不写
- 只给模型当前需要的两段外部证据；不回灌已有 claims/结果
- 限制字数：statement ≤200，来源 ≤4000，prompt 结论 ≤80
- 证据不足就 `none`，不要为了产出硬凑
- `explored` 记录试过的实体对，避免死循环
- 允许扩展实体，但扩展后继续推理才能产生新知识
- 多跳也用一张 v8 DAG：两条 claim → v7 图 → script 写 `derived_from`，不能冒充外部来源

**指标**

| 指标 | 定义 | 当前实测 |
|---|---|---|
| `source_coverage` | 有来源的 claim / 全部 claim | **1.0** |
| `claim_yield` | claim / 尝试的实体对 | **0.2**（1/5） |
| `no_relation_rate` | 证据不足拒绝 / 尝试对 | **0.8**（4/5） |
| `statement_chars` | claim 语句长度 | 24 字（上限 200） |
| `entity_growth` | 扩展新增实体 / 周期 | 2 个（3→5） |
| `derived_ratio` | derived_from claim / 全部 claim | 待真实链形成 |

## 5. 参考示例

| 场景 | 文件 |
|---|---|
| 单图串行 | `v7/README.md`、`v8/configs/v7_core/stage_all.json` |
| 多图并行 + 合并 | `v8/configs/v7_core/dag_parallel.json` |
| 外部资料并行汇总 | `v8/configs/use/research_brief.json` |
| SQLite → 结论 | `v8/configs/use/sqlite_report.json` |
| script → 模型 → judge | `v8/configs/use/script_judge.json` |
| 开放世界推理 | `v9/demo.py`、`v9/README.md` |

## 6. 自评记分卡（真实运行）

| 层 | 场景 | 结果 |
|---|---|---|
| v7 | 3 边串行 | 10.83s，3/3 成功 |
| v8 | 3 个 v7 子图并行 | 4.0s，2.7× |
| v8 | 3 个示例 DAG | 全部 ok，契约 3/3，输出 108/29/19 字 |
| v9 | Wikipedia 推理 | 5 实体/5 来源/1 claim；来源覆盖 1.0，claim 产率 0.2 |

## 7. 反模式

- 单链任务硬上 v8，只为“看起来并行”
- 分支剪掉后还用 `join:"all"`，下游永远等
- 输出不设上限，结果越写越长
- v9 没有来源就写 claim，或把已有结论塞回 prompt
- 用缓存掩盖慢/不稳定（当前策略明确不用）
