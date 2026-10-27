# 全量 DSH session：质量评分 + ELO 排行

扫描 `~/.dsh/session-manager/state.json` 中的全部 session，抽取最终结论，打确定性质量分，再两两对局算 Elo。

## 入口

```bash
cd /mnt/c/Users/lumin/Downloads/agent

# 抽样测试
python3 v8/configs/session_elo/run.py --limit 50

# 全量
python3 v8/configs/session_elo/run.py
```

## DAG

```text
scan    layer0  遍历 state.json + 所有 session.v4.jsonl.zstd
                 → v8/data/session_elo/sessions.json
score   layer1  确定性质量分 → scored.json
elo     layer2  两两对局（全对局 3 轮，K=16）→ ranking.json
report  layer3  Markdown / JSON 排行
```

## 质量分

| 维度 | 权重 | 依据 |
|---|---:|---|
| 内容量/具体性 | 30% | 长度、数字、代码块、路径 |
| 与用户诉求匹配 | 25% | 结论与 request 的字符 bigram 重合 |
| 结构 | 20% | 标题、列表 |
| 行动性 | 15% | 下一步/建议/风险/结论等词 |
| 完成状态 | 10% | turn/end completed |

## ELO

- 初始 1500，K=16
- 质量分差 ≤ 0.01 记为平局，否则高分为胜
- 3 轮全对局（每轮打乱顺序），降低固定顺序偏差
- 样本：所有有最终 assistant 文本的 session；空结论不参赛

## 输出

| 文件 | 内容 |
|---|---|
| `reports/session_elo_ranking.md` | 人看的排行：Top 30 / Bottom 10 / 样本分布 |
| `reports/session_elo_ranking.json` | 全部排名与分项分 |
| `v8/data/session_elo/sessions.json` | 扫描原始数据（已脱敏，未提交） |
| `v8/data/session_elo/scored.json` | 打分结果 |
| `v8/data/session_elo/ranking.json` | Elo 结果 |

## 注意

- 这是**启发式排序**，不是 LLM 语义评判。
- 长度、代码、路径会拉高分数，存在偏置。
- 空结论/被中断 session 会被压低；看排名时要区分“数据缺失”和“回答差”。
- 网关恢复后可加一层 LLM judge，对 Top/Bottom 做语义复核。
