# session + 结论文档 质量评判 DAG

输入**原始 DSH session** 和**结论文档**，输出评分、判定、发现、建议。

## 入口

```bash
cd /mnt/c/Users/lumin/Downloads/agent

python3 v8/configs/session_doc_judge/run.py \
  --session session-adf80908-d20f-4ffa-b00f-952b9b4bf15d \
  --doc reports/session_adf80908-d20_report.md
```

## DAG

```text
read_session   layer0  session_id → 原始 session JSON
read_doc       layer0  doc_path → 文档 JSON
├─ structure   layer1  标题层级/空章节/占位符/摘要/代码占比
├─ coverage    layer1  6 个复盘维度是否非空
└─ faithfulness layer1 与 session 原文重合度 + session id/source
merge          layer2  加权评分 + 判定 + 发现 + 建议
```

## 评分

- 覆盖 35%
- 结构 25%
- 忠实/可追溯 25%
- 可读性 15%

判定：`good >= 0.75`，`needs_work >= 0.55`，否则 `bad`。

## 输出

- `reports/session_doc_eval_<slug>.md`
- `reports/session_doc_eval_<slug>.json`

## 已测

对 `reports/session_adf80908-d20_report.md`：

```text
覆盖 0.50 (3/6)
结构 0.23
忠实 0.60
可读 0.70
总分 0.488 → bad
```

主要问题：3 个空/占位章节、原文标题未降级、缺结论先行、无 session id/source。
