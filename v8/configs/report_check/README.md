# 通用汇报格式检查 DAG

输入一份 Markdown 汇报，输出结构化判定：`pass / score / issues / suggestions / missing_sections / summary`。

## 用法

```bash
cd v8

# 直接跑
python -m dag8 run configs/report_check/dag.json \
  --seed "$(cat report.md)" --workers 3 --rpm 3 --json

# 或脚本
./configs/report_check/run_check.sh report.md --json
```

示例：

```bash
python -m dag8 run configs/report_check/dag.json \
  --seed "$(cat configs/report_check/examples/bad.md)" --workers 3 --rpm 3
```

## 示例

- `examples/bad.md`：故意空泛，预期 `pass=false`（实测 1.3 分，7 条问题）
- `examples/structured.md`：结构完整、数字标了样本/来源/验证状态，用作对照；
  但检查器仍可能因“证据不足/结论越级”拒绝（一次完整运行 6.3，未过 6.5），这是有意的严格

## 结构

```
report ─┬─ metrics   (command，确定性指标)
        ├─ structure (pi，结构评审)
        ├─ evidence  (pi，证据评审)
        └─ clarity   (pi，表达评审)
                └── verdict (pi，汇总判定)
```

- `metrics`：纯本地脚本，检查章节、数字、长句、术语、模糊词
- 三个评审并行；`verdict` 汇总加权评分并给可执行建议
- 契约：三个评审和终审都必须是 JSON；格式错自动 repair 一次

## 判定口径

- `score`：指标 25% + 结构 30% + 证据 20% + 表达 25%，一位小数
- `pass=true`：`score >= 6.5` 且结论 / 证据 / 下一步都不缺失
- 证据评审只查**标注是否完整**（基线/时间/样本/来源），不因样本少扣分；单次写“单次”即可
- `issues`：合并去重，按严重度排序
- `suggestions`：最多 5 条，要求可执行

## 确定性指标

`check_report.py` 从 stdin 读正文，输出：

```json
{
  "chars": 1000,
  "headings": ["结论", "下一步"],
  "numbers": 6,
  "sections": {"conclusion": true, "evidence": true, "risks": false,
               "next_steps": true, "background": true, "method": true},
  "first_screen_conclusion": true,
  "long_sentences": 1,
  "jargon": ["DAG", "RPM"],
  "vague": ["大概"],
  "empty": false
}
```

要扩充术语/模糊词，改脚本顶部的 `JARGON` / `VAGUE` 列表。

## 成本与速度

- 4 次 pi 调用（三评审 + 终审）+ 1 次本地命令
- 免费网关建议 `--workers 3 --rpm 3`，通常 1–3 分钟；模型慢时会更久
- 结果可缓存：同一份报告 + 同一模型，第二次直接命中

## 注意

- 这是**格式与表达**检查，不判断事实是否正确；事实错误仍需人工或专门核查
- LLM 评审有波动；`pass=false` 不等于报告一定差，`pass=true` 也不等于事实无误
- 想把阈值/权重改成团队规范，直接改 `dag.json` 里 `verdict` 的 prompt
