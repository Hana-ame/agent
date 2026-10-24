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
