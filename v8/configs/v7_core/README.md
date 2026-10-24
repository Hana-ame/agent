# v7 是 DAG 本体，v8 是组织层

- `stage_*.json`：普通 v7 DAG，负责任务内部的标签匹配与执行
- `dag_parallel.json`：v8 把 3 个 v7 子图当节点，并行组织；`result_tag=p1`
- `dag.json`：3 个 v7 子图并行 + 1 个 v7 合并

```bash
cd v8
python -m dag8 run configs/v7_core/dag_parallel.json --workers 3 --rpm 3 --json
python -m dag8 run configs/v7_core/dag_parallel.json --workers 1 --rpm 3 --json
```

实测（2026-10-25，短 v7 子图，每图一次调用）：

| workers | 墙钟 | 子图耗时 |
|---|---:|---|
| 3 | **4.0s** | 3.2 / 3.5 / 3.9s |
| 1 | **10.8s** | 3.5 / 3.6 / 3.5s |

并行 **2.7×**。这验证了定位：v7 负责“图怎么跑”，v8 负责“多张图怎么组织、并行、合并、落库”。

## 补充对照（2026-10-25）

同一批 3 个短 v7 边/子图：

| 执行方式 | 墙钟 | 说明 |
|---|---:|---|
| v7 本体串行（`stage_all.json`） | **10.83s** | 3 条边顺序跑 |
| v8 组织并行（`dag_parallel.json`，workers=3） | **4.0s** | 3 个 v7 子图同波次 |
| v8 组织 + 缓存第一次 | 7.87s | 网关慢窗口，3 个子图未命中 |
| v8 组织 + 缓存第二次 | **0.28s** | 3 个子图全部命中 |

结论：v7 本体负责单图执行；v8 组织层带来并行（约 2.7×）和跨 run 缓存（约 28×）。
并行倍率受单支慢调用影响，缓存倍率只在“同配置重复跑”成立。
