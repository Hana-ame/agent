# DAG Framework & Scripts

## 目录结构

```text
.
├── dag/                        # DAG 核心框架（支持 import dag）
│   ├── __init__.py             # 暴露 dag.load(), dag.step(), dag.contents, dag.edges, DAG
│   ├── engine.py               # DAG 核心执行引擎与单步调度
│   ├── schema.py               # models / contents / edges / tools 数据结构与 JSON 持久化
│   ├── matcher.py              # 标签匹配：fn 边只认精确 tag，LLM 边支持 N-gram/向量模糊
│   ├── default_processor.py    # 默认处理器（N-in-1-out 拼凑与干净上下文自主路由）
│   ├── llm.py                  # OpenAI 兼容客户端（支持流式与重试）与本地 Mock 客户端
│   ├── meta.py                 # 元图（Meta-DAG）动态修改器
│   └── builder.py              # 从自然语言或 JSON 描述构建 DAG
├── scripts/                    # 现有使用 dag 的业务脚本
│   ├── s1/                     # Stage1st (S1) 论坛帖子抓取与 AI 情报摘要
│   │   ├── stage1st_ai.py              # S1 157 归墟板块近 24h AI 情报摘要 DAG
│   │   ├── stage1st_stability.py       # S1 分类与摘要稳定性测试脚本
│   │   └── stage1st_crawler_digest.py  # S1 157 爬虫、Python 交错与时间线汇总脚本
│   └── cf/                     # Cloudflare 子域流量拉取、校验与日报
│       ├── run_cf_dag.py               # CF 子域×每日流量 + Web Analytics 页面浏览量 DAG
│       ├── cf_fns.py                   # CF 数据拉取与三道闸校验函数
│       └── models.json                 # CF 脚本免 Key 网关配置
├── tests/                      # 框架回归测试（标签匹配串数据 / fn 边契约）
├── .gitignore                  # 忽略真实密钥与所有脚本运行产物（cf_report.md, out/ 等）
├── models.json.example         # 模型配置模板（复制为 models.json 填写密钥后使用）
└── pyproject.toml              # 包构建配置
```

## 快速使用 `dag` 框架

```python
import dag

# 1. 加载（支持 JSON 文件路径、包含 models/contents/edges.json 的目录、JSON 字符串或 dict）
dag.load("path/to/dag.json")

# 2. 单步执行
dag.step()

# 3. 查看或读写当前内容池与边集合
print(dag.contents)
print(dag.edges)
```

也支持实例式调用：

```python
import dag

d = dag.load("path/to/dag.json")
d.step()
print(d.contents)
print(d.edges)
```

## 运行现有脚本

```bash
# 1. Stage1st 157 归墟板块近 24h AI 情报摘要（产物默认输出至 scripts/s1/out/stage1st_ai_24h.md）
python3 scripts/s1/stage1st_ai.py

# 2. Stage1st 稳定性测试
python3 scripts/s1/stage1st_stability.py classify 4

# 3. Cloudflare 子域×每日流量统计 + Web Analytics 页面浏览量（排除 Bot）
#    （产物默认输出至 scripts/cf/cf_report.md）
python3 scripts/cf/run_cf_dag.py
```
