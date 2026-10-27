# ELO 裁判 DAG 结果

> 生成：2026-10-28 05:03:33　选手：12　对局：32　有效票：32

## 方法

- 样本：top / bottom / 随机三段
- 裁判后端：local
- 裁判：逐对判 A/B/tie，票直接更新 Elo（K=32，初始 1500）
- 这是裁判票的结果，不是启发式质量分

## 排名

| # | Elo | 质量分(参考) | Session | 标题 | 战绩 |
|---:|---:|---:|---|---|---|
| 1 | 1583 | 0.90 | `9847d2c0-f3c0-47df-8c21-` | 你是只读代码体检员。任务： | 6W/0L/1T |
| 2 | 1569 | 0.92 | `d6069d0c-5b5b-4962-abb3-` | 只读调研任务。禁止使用 web_se | 5W/0L/1T |
| 3 | 1546 | 0.92 | `45d88cac-1a06-43d5-9f6f-` | 【任务性质】只读架构体检。 | 3W/0L/1T |
| 4 | 1530 | 0.42 | `session-9bf5f085-1a7d-43` | 查schedule-catalog来源 | 3W/1L/0T |
| 5 | 1528 | 0.91 | `7be3a1a1-232f-4939-a56f-` | You are doing READ-ONLY research | 3W/1L/1T |
| 6 | 1503 | 0.05 | `session-30a38955-791c-47` | PR-tag标签collection搜索 | 2W/2L/0T |
| 7 | 1501 | 0.37 | `session-ad3a971a-fa24-44` | 提交 reflog 第四种用法 | 3W/3L/0T |
| 8 | 1488 | 0.29 | `session-ca156ce7-4356-43` | 清理 peerdrive 废弃分支 | 2W/3L/0T |
| 9 | 1453 | 0.00 | `session-d30b6637-79cf-44` |  | 0W/3L/1T |
| 10 | 1453 | 0.00 | `8a6b2a61-f3fb-4f62-856c-` | 你在审计一个本地个人知识库 | 0W/3L/2T |
| 11 | 1428 | 0.00 | `acc7b1fe-39ed-45f4-bd33-` | 你在审计一个本地个人知识库 | 0W/5L/1T |
| 12 | 1418 | 0.00 | `51ecfcef-8719-4611-9524-` | 你是交叉取证复核员。工作目 | 0W/6L/2T |

## 裁判明细

| 对局 | A | B | 胜者 | 理由 |
|---|---|---|---|---|
| match_1 | `d6069d0c-5b5b-4962` | `51ecfcef-8719-4611` | A | local score |
| match_2 | `d6069d0c-5b5b-4962` | `45d88cac-1a06-43d5` | tie | local score |
| match_3 | `d6069d0c-5b5b-4962` | `session-30a38955-7` | A | local score |
| match_4 | `45d88cac-1a06-43d5` | `acc7b1fe-39ed-45f4` | A | local score |
| match_5 | `45d88cac-1a06-43d5` | `8a6b2a61-f3fb-4f62` | A | local score |
| match_6 | `7be3a1a1-232f-4939` | `9847d2c0-f3c0-47df` | tie | local score |
| match_7 | `7be3a1a1-232f-4939` | `session-ad3a971a-f` | A | local score |
| match_8 | `d6069d0c-5b5b-4962` | `7be3a1a1-232f-4939` | A | local score |
| match_9 | `9847d2c0-f3c0-47df` | `acc7b1fe-39ed-45f4` | A | local score |
| match_10 | `9847d2c0-f3c0-47df` | `session-30a38955-7` | A | local score |
| match_11 | `9847d2c0-f3c0-47df` | `51ecfcef-8719-4611` | A | local score |
| match_12 | `session-30a38955-7` | `51ecfcef-8719-4611` | A | local score |
| match_13 | `session-30a38955-7` | `session-d30b6637-7` | A | local score |
| match_14 | `9847d2c0-f3c0-47df` | `session-ad3a971a-f` | A | local score |
| match_15 | `session-ad3a971a-f` | `acc7b1fe-39ed-45f4` | A | local score |
| match_16 | `session-ad3a971a-f` | `session-ca156ce7-4` | A | local score |
| match_17 | `session-ad3a971a-f` | `session-9bf5f085-1` | B | local score |
| match_18 | `9847d2c0-f3c0-47df` | `session-9bf5f085-1` | A | local score |
| match_19 | `session-9bf5f085-1` | `51ecfcef-8719-4611` | A | local score |
| match_20 | `session-ca156ce7-4` | `acc7b1fe-39ed-45f4` | A | local score |
| match_21 | `45d88cac-1a06-43d5` | `session-ca156ce7-4` | A | local score |
| match_22 | `9847d2c0-f3c0-47df` | `session-ca156ce7-4` | A | local score |
| match_23 | `d6069d0c-5b5b-4962` | `acc7b1fe-39ed-45f4` | A | local score |
| match_24 | `session-ca156ce7-4` | `8a6b2a61-f3fb-4f62` | A | local score |
| match_25 | `d6069d0c-5b5b-4962` | `8a6b2a61-f3fb-4f62` | A | local score |
| match_26 | `acc7b1fe-39ed-45f4` | `8a6b2a61-f3fb-4f62` | tie | local score |
| match_27 | `session-ad3a971a-f` | `51ecfcef-8719-4611` | A | local score |
| match_28 | `7be3a1a1-232f-4939` | `51ecfcef-8719-4611` | A | local score |
| match_29 | `8a6b2a61-f3fb-4f62` | `51ecfcef-8719-4611` | tie | local score |
| match_30 | `7be3a1a1-232f-4939` | `session-d30b6637-7` | A | local score |
| match_31 | `session-9bf5f085-1` | `session-d30b6637-7` | A | local score |
| match_32 | `51ecfcef-8719-4611` | `session-d30b6637-7` | tie | local score |
