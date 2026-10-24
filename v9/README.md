# v9：开放世界知识推理层

v7 是 DAG 本体，v8 是组织层，v9 负责**知识**：

- 实体 / 关系 / 来源 / 推理链，而不是一次 run 的临时 contents
- 每条 claim 必须能追到 `source_url`（外部证据）或 `derived_from`（推理来源）
- 推理循环：现有知识 → 候选实体对 → 抓外部证据 → 判定关系 → 写回
- 明确允许“证据不足 → 无关系”，不为了产出而编造

## 结构

```text
v9/
├── knowledge.py   # entities / claims / sources / explored
├── infer.py       # InferenceEngine: 选实体对 → 抓证据 → 生成 → 校验来源 → 入库
└── demo.py        # Wikipedia 真实演示
```

## 跑

```bash
cd /mnt/c/Users/lumin/Downloads/agent
python -m v9.demo
python -m pytest v9/tests -q
```

## 真实结果（2026-10-25）

输入实体：`Directed acyclic graph` / `Topological sorting` / `Blackboard system`。

- 抓了 3 个 Wikipedia 来源
- 2 对判定“证据不足，无关系”
- 1 条新知识入库：

```text
Directed acyclic graph --uses--> Topological sorting
confidence 0.95
source: https://en.wikipedia.org/api/rest_v1/page/summary/Directed_acyclic_graph
```

`explored` 记录了“试过但没有关系”的实体对，避免每轮重复问同一个问题。

## 当前边界

- 只做 1-hop 实体对，没有多跳链式推理
- 没有矛盾检测（同一对出现相反 claim 时不会自动降权）
- 来源只保存 URL + 摘要，没有精确到句子/段落的证据定位
- 实体列表目前是人工 seed；还没有“从外部页面自动扩展实体”
