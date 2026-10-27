# 专门的 ELO 裁判 DAG

不再用启发式质量分直接排序，而是：**选样本 → 生成对局 → 裁判逐对判 A/B/tie → 裁判票算 Elo**。

## 入口

```bash
cd /mnt/c/Users/lumin/Downloads/agent

# 本地裁判（验证 DAG；用质量分模拟裁判，不调模型）
python3 v8/configs/session_elo_judge/run.py --backend local --n 4

# 真实 LLM 裁判（网关可用时）
python3 v8/configs/session_elo_judge/run.py --backend http --n 4
```

## DAG

```text
select    layer0  从 v8/data/session_elo/scored.json 选 top/bottom/随机
                  生成每人约 3 场的跨组对局 → pairs JSON
judge     layer1  v7 图：逐对判 A/B/tie
                  输出 {"votes":[{"pair_id","winner","reason"}]}
aggregate layer2  用裁判票更新 Elo（K=32，初始 1500）→ 报告
```

## 两种裁判后端

| backend | 行为 |
|---|---|
| `local` | 用已有质量分模拟裁判，验证 DAG 和 Elo 更新；**不是真实评判** |
| `http` | 走 `laptop-8000.moonchan.xyz` 的真实模型裁判；网关 502 时会失败 |

## 输出

- `reports/session_elo_judge.md`
- `reports/session_elo_judge.json`

## 已测（local）

```text
select     0.1s  19367 字符
judge      0.0s   1191 字符
aggregate  0.1s    144 字符
选手 12 / 对局 32 / 有效票 32
```

真实 LLM 裁判当前被 moonchan 502 阻塞；网关恢复后 `--backend http` 即可。
