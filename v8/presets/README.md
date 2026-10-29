# presets：预设检查 DAG

对一个对话 session，挂 N 条**预设**。每条预设 = 一段完整 system prompt + 一个专属 LLM，
检查当前会话状态是否满足该预设；所有预设**每次都全部并行检查**，再确定性聚合。

```text
state ─→ check_clarify (pi) ─┐
state ─→ check_hallucination  ┼→ aggregate (script) ─→ result
state ─→ check_off_topic   ──┘
```

- 预设文件：`presets/examples/presets/*.json`，字段 `id / name / system_prompt / on_hit`
- 会话状态模式：`full`（全文+元信息）、`last`（最近 N 条，默认 1）、`mainline`（主轴）
- 每个 pi 节点 prompt = 预设 system prompt + 状态 + “只输出 JSON”
- 聚合是确定性 script（`scripts/aggregate.py`），不调 LLM 总结

## 跑

```bash
cd /mnt/c/Users/lumin/Downloads/agent/v8

# 不调模型，只看生成的 DAG
python -m presets --presets presets/examples/presets \
    --conv presets/examples/conv_example.json --state mainline --dry

# 真实检查（并行跑所有预设）
python -m presets --presets presets/examples/presets \
    --conv presets/examples/conv_example.json --state mainline

# 网关慢/离线时，用假 agent 跑通全流程
python -m presets --presets presets/examples/presets \
    --conv presets/examples/conv_example.json --state mainline --mock

# 调大每个 pi 节点的超时（默认 300s）
python -m presets ... --timeout 600
```

输出（`summary.json`）：

```json
{
  "checked": 3,
  "satisfied": 2,
  "hits": [
    {"preset": "clarify", "satisfied": true, "score": 5, "reason": "…"},
    {"preset": "hallucination_risk", "satisfied": true, "score": 3, "reason": "…"}
  ],
  "all": [ ... ]
}
```

## 设计对应

| 你的描述 | 这里 |
|---|---|
| 多个预设 | `presets/examples/presets/*.json` |
| 预设是一段判断条件，也是 LLM | 每条预设 = 一个 pi 节点，system prompt 就是预设 |
| 检查 session 当前状态 | `state.py` 把会话渲染成全文本/最近一条/主轴 |
| 对所有的都检查 | 所有预设节点同一 layer 并行，无短路 |
| 命中干啥 | 目前只记录 + 汇总；`on_hit` 字段留作后续动作口子 |

## 自定义

- 新建预设：往 presets 目录加一个 JSON，写清楚“什么算满足”。
- 状态范围：`--state full|last|mainline [--last N]`。
- 模型：`--provider laptop --model sensenova-6.8-flash-lite`（默认即该组合）。

## 测试

```bash
cd /mnt/c/Users/lumin/Downloads/agent/v8
python -m pytest tests/test_presets.py -q
```
