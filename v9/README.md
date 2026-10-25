# v9：开放世界知识推理层

v7 是 DAG 本体，v8 是组织层，v9 负责**知识**：

- 实体 / 关系 / 来源 / 推理链，而不是一次 run 的临时 contents
- 每条 claim 必须能追到 `source_url`（外部证据）或 `derived_from`（推理来源）
- 推理循环：现有知识 → 候选实体对 → 抓外部证据 → 判定关系 → 写回
- 明确允许“证据不足 → 无关系”，不为了产出而编造

## 硬约束

- **不缓存**：`V8Executor` 固定 `use_cache=False`；v8 的 run 轨迹只作调试，不复用
- **契约保留**：`v9/configs/gen.json` 的 graph 节点带 v8 `expect`，输出必须是含
  `relation/statement/confidence/evidence_url` 的 JSON；不合格边 error，不写知识
  （graph 节点不支持 repair，失败即失败）
- **只存知识**：长期数据只有 `claims / sources / explored`；原始结果不写回知识
- **限制字数**：claim statement ≤ 200 字；外部来源 ≤ 4000 字；prompt 要求结论 ≤ 80 字
- **限制结果透露**：推理时只给两段外部证据，不给已有 claims / 最终结果；
  超长、无来源、来源不在抓取集合里的 claim 一律拒绝
- **来源归一化**：模型引用 Wikipedia 人类页面时，映射到实际抓取的 API summary URL；
  仍不会接受任意外部链接

## 结构

```text
v9/
├── knowledge.py   # entities / claims / sources / explored
├── infer.py       # InferenceEngine: 选实体对 → 抓证据 → 生成 → 校验来源 → 入库
├── executor.py    # V8Executor：生成走 v7 图 + v8 graph 节点，抓取走 v8 http 节点
├── expand.py      # 从 Wikipedia 外链扩展实体，打破人工 seed 封闭集
├── configs/       # v8 的 gen/get 两个小 DAG
├── graphs/gen.json# v7 单边 LLM 图
└── demo.py        # Wikipedia 真实演示
```

## 执行层复用

v9 不自己造调度：

- `generate(prompt)` → v8 跑 `configs/gen.json`，里面是一个 graph 节点，再调用 v7 `graphs/gen.json` 的单边 LLM 图
- `fetch(url)` → v8 跑 `configs/get.json`，用 v8 的 http 节点
- 两者都复用 v8 的 sqlite 轨迹、`use_cache=True` 缓存、超时/重试

## 跑

```bash
cd /mnt/c/Users/lumin/Downloads/agent
python -m v9.demo
python -m pytest v9/tests -q
```

## 多跳推理

```python
res = engine.derive()   # A --p--> B + B --q--> C ⇒ A --r--> C
```

- 只使用两条已有 claim 的 statement，不重新抓外部页面
- 新 claim 无 `source_url`，但 `derived_from=[c1_id, c2_id]`
- 推不出就 `none`，不写；重复的 A→C 不再推
- 当前真实 demo 只有一条外部 claim，尚未形成真实链；多跳先在单测覆盖

## 实体扩展

```python
from v9.expand import expand
added = expand(knowledge, executor.fetch, "Directed acyclic graph", limit=2)
```

从 Wikipedia `prop=links` 抓外链，去重、跳过已有实体，写回 `entities`。
实测一次扩展：新增 `Academic Press`、`Acyclic dependencies principle`；
知识库从 3 个实体增长到 5 个，来源数 5。

## 真实结果（2026-10-25）

输入实体：`Directed acyclic graph` / `Topological sorting` / `Blackboard system`。

- 抓了 3 个 Wikipedia 来源
- 2 对判定“证据不足，无关系”
- 1 条新知识入库：

```text
Directed acyclic graph --applies_to--> Topological sorting
confidence 1.0
source: https://en.wikipedia.org/api/rest_v1/page/summary/Topological_sorting
```

`explored` 记录了“试过但没有关系”的实体对，避免每轮重复问同一个问题。

## 当前边界

- 只做 1-hop 实体对，没有多跳链式推理
- 没有矛盾检测（同一对出现相反 claim 时不会自动降权）
- 来源只保存 URL + 摘要，没有精确到句子/段落的证据定位
- 实体扩展是 1 跳外链，按页面顺序取前 N 个；还没有按相关性筛选
- 扩展后的新实体需要继续跑推理循环才会产生新知识
