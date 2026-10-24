# evolve：受约束的 DAG 演化 + 人工评价候选

一条最小但完整的闭环：

```text
Genome（dag.json + workers + 变异谱系）
  → mutate（prompt / thinking / contract / critique 节点 / workers）
  → run（dag8 执行，落独立目录与 sqlite）
  → 自动检查（长度 / 标题 / 关键词 / 禁用词）
  → 人工评价包（review.md + ratings.json）
  → select（human / auto / combined）
```

**人工评价是一等公民**：每个候选都有独立输出、评分位和备注；没有可信自动指标时，
`--by human` 仍然可以选出候选。自动检查只做便宜的确定性检查，不替代人。

## 快速开始

```bash
cd /mnt/c/Users/lumin/Downloads/agent/v8

# 1) 真实候选：troll_notes 任务，population=1（首跑会调模型；命中缓存则很快）
python3 -m evolve run --task evolve/tasks/troll_notes.json --population 1 \
  --out data/human-eval/troll-real

# 2) 生成多个候选：population=3；--workers 只影响 base，变异可能改 workers
python3 -m evolve run --task evolve/tasks/troll_notes.json --population 3

# 3) 不跑模型，只从历史 run 生成人工评价包
python3 -m evolve from-runs --db data/dag8.sqlite3 --last 12 \
  --out data/human-eval/from-runs

# 4) 离线自检：假 agent，验证评价包链路
python3 -m evolve run --task evolve/tasks/demo.json --population 3 --mock \
  --out data/human-eval/demo-mock
```

## 答案泄漏惩罚（分数 ×0.1）

规则：**如果候选（DAG prompt 或输出）里出现“仅在参考答案中拥有、源材料里没有”的信息，
自动分直接乘以 0.1。**

实现：[`evolve/leakage.py`](leakage.py)。它只做保守的逐字匹配：

1. 把参考答案拆成标题/句子/表格单元；
2. 去掉 Markdown 与标点后，长度 ≥ 8 的片段若在 source 中找不到，记为 answer-only；
3. 候选的 **prompt** 或 **output** 只要包含任一 answer-only 片段（≥8 字），即判泄漏；
4. `score = score_raw × 0.1`，并把命中片段写进 `review.md` / `review.json`。

评测器读参考答案是允许的；候选 DAG 不读。给命令加 `--source` 与 `--answer` 才会启用：

```bash
python3 -m evolve from-runs --db data/dag8.sqlite3 --last 12 \
  --source /tmp/troll_source.txt \
  --answer /home/lumin/Downloads/troll_tea_party_notes.md \
  --out data/human-eval/from-runs-leak
```

`review.md` 会显示 `10.0（原 100.0） | 是 ×0.1`，并列出命中的 answer-only 片段。
缺 `--source` 或 `--answer` 时明确标“未启用”，不会假装查过。

`--source` 必须是 DAG 实际看到的完整 source（header + 元信息 + 转写）。只给转写会把
标题/URL 误判成 answer-only——`from-runs` 的实测就踩过这个坑。

局限：它抓逐字泄漏（标题、表头、勘误表、原句），不抓同义改写；同义改写需要语义判据，
不能用一个便宜的确定性检查冒充。

## 人工评分

打开 `review.md` 看候选全文，然后二选一：

```bash
# 命令行评分
python3 -m evolve rate --run data/human-eval/troll-real \
  --candidate c1e19e7a961e --score 4 --note "结构完整，开篇略长"

# 或者直接编辑 ratings.json：
# {"ratings": {"<id>": {"score": 4, "note": "..."}}}
```

选择：

```bash
python3 -m evolve select --run data/human-eval/troll-real --by human
python3 -m evolve select --run data/human-eval/troll-real --by auto
python3 -m evolve select --run data/human-eval/troll-real --by combined
```

- `human`：必须所有候选都有 1~5 分；按人工分排，自动分做 tie-break。
- `auto`：只看确定性检查分，适合先粗筛。
- `combined`：`0.7 × 人工分归一化 + 0.3 × 自动分`。

产物：

```text
data/human-eval/<run>/
  task.json         任务描述
  genomes.json      候选（完整 DAG 配置 + 谱系）
  candidates/<id>/
    config/dag.json 候选实际跑的配置
    output.md       候选输出
    record.json     状态/耗时/自动检查
    run.sqlite3     dag8 运行轨迹
  review.md         人看的评分包（含全文）
  review.json       机读评价包
  ratings.json      人工评分模板 / 结果
  selection.json    最终选择与排名
```

## 变异算子

| 算子 | 改什么 | 说明 |
|---|---|---|
| `prompt_style` | final prompt 追加风格约束 | 克制 / 现场感 / 管理者视角 / 更短 |
| `prompt_requirement` | final prompt 追加硬要求 | 过秦论/三苏、3.2 五阶段、3.4 三列八行 |
| `thinking` | `pi.thinking` | `off` ↔ 默认 |
| `contract` | `expect` + `repair` | 给 final 加长度/标题/mermaid 约束 |
| `critique_node` | 拓扑 | 在 final 前插 `self_critique` 边并接进 final 输入 |
| `workers` | 运行参数 | 1 ↔ 2 |

`prompt_*` 只追加、不改原文；`critique_node` 只在 final 的输入含 `draft` 时生效；
任何变异都不改 `evolve/` 或 `dag8/` 代码。

## 设计边界

- **Run 目录隔离**：候选配置复制到 `candidates/<id>/config/`，`command` 节点的相对路径
  （如 `python3 trim_md.py`）仍成立。
- **评测器不在候选里**：候选只改 DAG 配置；`evolve` 的检查/评分/选择代码是冻结层。
- **不自动改代码**：`critique_node` 只改图；节点实现、工具、执行器代码不进入变异空间。
- **人工闸门**：`--by human` 是默认选择方式；`--by auto` 只适合粗筛，不能替代人。
- **可复现**：`genomes.json` + 候选配置 + `run.sqlite3` + `review.json` 足以复跑与追责。

## 已验证

```bash
python3 -m pytest tests/test_evolve_human.py tests/test_evolve_leakage.py -q  # 11 passed
python3 -m evolve from-runs --db data/dag8.sqlite3 --last 12
python3 -m evolve run --task evolve/tasks/troll_notes.json --population 1 \
  --out data/human-eval/troll-real
```

`from-runs` 不调模型；`--mock` 也不调模型。真实候选生成会走 `dag8`，费用/时延取决于任务图。
