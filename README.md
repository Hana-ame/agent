# DAG Framework & Scripts

## 版本结构

```text
v7/                      # DAG 本体（canonical，可 import v7；v7.step()/contents/set）
v8/                      # 组织层（多张 v7 图并行、合并、契约、落库；不缓存）
v9/                      # 开放世界知识推理层（实体/关系/来源/推理循环）
v4/                      # VEA v4 历史设计备份（参考，不参与运行）
GUIDE.md                 # v7/v8/v9 使用方式、最佳实践、自评指标
docs/ARCHITECTURE.md     # 当前分层、去重决策、执行链、限制
docs/interfaces.html     # 接口表 + 可拖拽拓扑图
docs/interfaces.json     # 接口规范 JSON 源
v8/configs/dsh_report/   # DSH 历史 session 汇总报告（v7/v8/v9 示例）
v8/configs/dsh_session_report/ # 指定单个 DSH session 的复盘报告 DAG
v6/                      # VEA v6 历史设计备份（参考，不参与运行）
v8/configs/v7_core/      # “v7 本体 + v8 组织”可运行原型
dag/                     # 旧 import 路径（兼容 scripts/tests，不再作为新入口）
```

- v7 负责“图怎么跑”：标签匹配、边执行、DefaultProcessor、Meta-DAG
- v8 负责“多张图怎么组织”：波次并行、合并、契约、质量闸门、SQLite 轨迹；缓存关闭
- v9 负责“知识怎么增长”：外部证据、关系推理、来源与推理链
- 新代码优先 `import v7`；v8 的 graph 节点也优先加载 `v7`，失败再回退 `dag`


## 目录结构

```text
.
├── dag/                        # 旧 import 路径（兼容；canonical 是 v7/）
│   ├── __init__.py             # 旧入口（兼容）；新代码用 v7/
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

## 快速使用 `v7` 本体

```python
import v7

# 1. 加载（支持 JSON 文件路径、包含 models/contents/edges.json 的目录、JSON 字符串或 dict）
v7.load("path/to/dag.json")

# 2. 单步执行
v7.step()

# 3. 查看或读写当前内容池与边集合
print(v7.contents)
print(v7.edges)
```

也支持实例式调用：

```python
import dag

d = v7.load("path/to/dag.json")
d.step()
print(d.contents)
print(d.edges)
```

## 测试

根目录已配置 `pytest.ini`，默认收集全部项目测试：

```bash
python -m pytest -q                 # 318 条：根 19 + v8 279（含 4 live）+ v9 20
python -m pytest -q -m "not live"   # 314 条：不碰模型
cd v8 && python -m pytest tests/ -q # 只跑 v8 的 279 条
python -m pytest v9/tests -q        # 只跑 v9 的 20 条
```

- v9 用 Wikipedia 摘要做真实外部证据；无网络时 demo 会失败
- `.venv/` 里的是第三方依赖测试，不属于本项目
- live 用例需要真实 pi；没有 pi 会直接 fail，不 skip

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
