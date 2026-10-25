# 可直接用的 3 个小 DAG

先建演示库：

```bash
cd v8
python configs/use/setup_sales.py
```

## 1. research_brief.json

两个 Wikipedia 页面并行抓取 → 合并成 3 条要点。

```bash
python -m dag8 run configs/use/research_brief.json --workers 2 --rpm 3 --json
```

实测：w1/w2 各 1.5s 并行，brief 18.6s，输出 3 条 DAG/拓扑排序要点。

## 2. sqlite_report.json

SQLite 查销量 Top5 → 模型写一句结论。

```bash
python -m dag8 run configs/use/sqlite_report.json --workers 2 --rpm 3 --json
```

实测：rows 0.0s，conclusion 9.1s，输出“华东销量最高，西南最低，整体呈下降趋势。”

## 3. script_judge.json

脚本统计字符/行数 → 模型改写 → judge 闸门。

```bash
python -m dag8 run configs/use/script_judge.json --workers 2 --rpm 3 --json
```

实测：stats 0.0s，rewrite 22.0s，输出“全文仅4词、30字符，分布于1行。”
注意 `stats.py` 的 words 按空白切分，中文会偏低；需要中文分词可自行替换。
