# 可复现的真实链路配置

这些配置用于复现 `STATUS.md` / 自检报告里的实机证据；不是日常模板。

## judge_gate.json

pi 产出后用 judge 打分，`min_score=0` 走通过路径：

```bash
cd v8
python -m dag8 run configs/live/judge_gate.json --workers 2 --rpm 3 --json
```

把 `judge.min_score` 改成 `10` 可复现拒绝路径（实测 6 分 < 10 被拒）。

## adaptive_8.json

8 条边同波次 + 1 条只依赖 `o2` 的第二波边：

```bash
python -m dag8 run configs/live/adaptive_8.json \
  --workers 8 --adaptive --keep-going --max-seconds 300 --json
```

网关过载时 `workers_used` 会从 `[8]` 降到 `[8,4]`；全成功时可能是 `[8,8]`。
这是过载观察，不保证每次都降。

## nested_inner.json / nested_outer.json

外层 graph 节点跑一张 v8 子图：

```bash
python -m dag8 run configs/live/nested_outer.json --workers 2 --rpm 3 --json
```

内层是“2 并行 + 1 合并”；`nested_inner.json` 单独跑也可以。

## 注意

- 都需要真实 `laptop` 网关；网关 429/530 时运行会失败
- 模型是 `deepseek-flash`，网关慢时会触发 timeout
- 这些配置的证据是单次观察，不是稳定性结论
