# Agent 工作区 · 项目总览

> 这不是一个单一项目：仓库里并存着一条 **Python DAG 框架线**、两个**实战脚本**、一个**可视化任务图 Web 应用（当前主线）**，以及一批用框架跑出来的**自评报告**。
> 本页是唯一入口——先看「项目地图」，再按需跳各自的 README。**动手改代码前先读 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)**（环境、开发循环、分层规则、测试门禁、真实踩坑）。

## 项目地图

| 目录 / 文件 | 是什么 | 状态 | 入口 |
|---|---|---|---|
| [taskpad/](taskpad/README.md) | 零依赖 Node.js 的 ComfyUI 风格节点图应用：项目 → 模块 → 接口 → 流程图，**网站只调度存储、各机器 worker 拉任务执行** | 🟢 **当前主线**（近期提交集中在这里） | [taskpad/README.md](taskpad/README.md) |
| [v7/](v7/README.md) · [v8/](v8/README.md) · [v9/](v9/README.md) | Python 标签 DAG 框架三层：v7 本体 / v8 组织执行 / v9 知识推理 | 🟡 稳定，按需维护 | [GUIDE.md](GUIDE.md) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| [scripts/](scripts/) | 框架的两个实战用例：S1 论坛 AI 情报摘要、Cloudflare 流量日报 | 🟡 可用（要 Key / 网络） | [scripts/s1/](scripts/s1/) · [scripts/cf/](scripts/cf/) |
| [reports/](reports/) | 用 v8/v9 跑 DSH 会话数据得到的产品：汇总报告、单 session 复盘、质量评判、ELO 排行 | ⚪ 生成物归档 | [reports/session_elo_ranking.md](reports/session_elo_ranking.md) |
| [v4/](v4/README.md) · [v6/](v6/README.md) · [dag/](dag/) | 历史版本与旧 import 兼容路径，**不参与运行** | 📦 只读参考 | 见各目录 README |

- 架构与去重决策：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)；接口规范：[docs/interfaces.json](docs/interfaces.json)、可拖拽拓扑图 [docs/interfaces.html](docs/interfaces.html)
- 开发指南（环境 / 开发循环 / 分层规则 / 提交门禁）：[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- 使用指南 / 最佳实践 / 自评指标：[GUIDE.md](GUIDE.md)

---

## 1. taskpad（当前主线）

**零依赖**（Node ≥ 18，不需要 npm install）的本地/内网 Web 应用。界面仿 ComfyUI：

- 数据模型：**图 → 节点（含递归子图）→ 边（接口契约 + 可执行接口测试）**
- 网站只做**调度 + 存储**：不执行任何命令；worker 从网站**拉取**待认领任务，在各自机器上执行后把 stdout / 成果文件**上传**回来
- 功能：运行图（拓扑排序）、单节点启动、定时任务、SSE 实时日志、Trace 历史、失联任务回收、Agent 网页工位

```bash
cd taskpad
node server.js          # 默认 http://127.0.0.1:8330（支持 --port/--host/--data-dir/--log）

# 另开终端：真实执行任务的节点（Python 标准库即可，可多开 / 部署到别的机器）
python3 worker.py --server http://127.0.0.1:8330 --worker-id my-node --root ..
```

> ⚠️ 任务命令会以当前用户权限在本机 shell 执行，**仅限本机/内网使用**。服务端结构、API 一览、常见问题见 [taskpad/README.md](taskpad/README.md)。

## 2. Python DAG 框架：v7 / v8 / v9

| 层 | 只负责 | 入口 | 有 CLI |
|---|---|---|---|
| **v7** | 标签 DAG 本体：边匹配、单图 `step()`、DefaultProcessor、Meta-DAG | `import v7`（`load/step/set/contents`） | 否，纯库 |
| **v8 / dag8** | 组织执行层：JSON 配置图、波次并行、契约校验、SQLite 轨迹 | `python -m dag8 run <config>` | 是（18 个子命令） |
| **v9** | 知识推理层：实体/关系/来源/推理链、多跳、实体扩展 | `CycleEngine` / `python -m v9.demo` | 否，Python API + demo |

依赖方向固定 **`v9 → v8 → v7`**，不允许反向 import。`dag/` 是 v7 之前的旧路径，仅供老脚本/tests 兼容。

```python
import v7
v7.load("graph.json")
v7.step()
print(v7.contents)      # tag -> 内容
v7.set("external", "外部处理结果")
v7.step()
```

```bash
cd v8
python -m dag8 demo                          # 离线 demo，不调模型
python -m dag8 run configs/demo.json --dry-run
python -m dag8 run configs/v7_core/dag_parallel.json --workers 2 --rpm 3
python3 -m v9.demo                           # 需要网络（Wikipedia 外部证据）
```

配置示例：v8 模板 `new` / `scaffold`；现成配置见 [v8/configs/](v8/configs/)（含 [v7_core/](v8/configs/v7_core/)、[use/](v8/configs/use/)、[dsh_report/](v8/configs/dsh_report/)、[session_elo/](v8/configs/session_elo/)）。

## 3. 实战脚本

```bash
python3 scripts/s1/stage1st_ai.py        # S1 157 归墟板块近 24h AI 情报摘要 → scripts/s1/out/
python3 scripts/cf/run_cf_dag.py         # CF 子域×每日流量 + Web Analytics 页面浏览量 → scripts/cf/cf_report.md
```

## 4. reports（自评产物）

v8/v9 跑 DSH 历史 session 数据得到的结论文件（**已提交**，约 2.2MB）：

- 汇总：[reports/dsh_sessions_report.md](reports/dsh_sessions_report.md)、[reports/dsh_session_report_latest.md](reports/dsh_session_report_latest.md)
- 单 session 复盘：[reports/session_adf80908-d20_report.md](reports/session_adf80908-d20_report.md)
- 质量评判 / ELO：[reports/session_elo_ranking.md](reports/session_elo_ranking.md)、[reports/session_elo_judge.md](reports/session_elo_judge.md)、人工核对 [reports/session_elo_judge.html](reports/session_elo_judge.html)

## 5. 仓库结构

```text
.
├── README.md                  # 本页：项目地图与总入口
├── GUIDE.md                   # v7/v8/v9 使用指南、最佳实践、自评指标
├── taskpad/                   # 当前主线：节点图 Web 应用（server.js / worker.py / public/ / src/ / test/ / docs/）
├── v7/                        # DAG 本体（canonical）
├── v8/                        # 组织/执行层：dag8/、configs/、evolve/、presets/、tests/
├── v9/                        # 知识推理层：knowledge/infer/cycle/expand + configs/
├── dag/                       # 旧 import 兼容路径（≈ v7 旧版全量拷贝）
├── v4/ · v6/                  # VEA 历史设计备份（只读参考；远端另有专门分支）
├── scripts/                   # s1/ 论坛情报、cf/ CF 流量、clean-local.sh 本地清理、ui.ps1
├── reports/                   # DSH session 报告与 ELO 排行（生成物，已入库）
├── docs/                      # DEVELOPMENT.md、ARCHITECTURE.md、interfaces.{html,json}
├── tests/                     # 根目录框架测试（19 条）
├── models.json.example        # 模型配置模板（复制为 models.json 填 Key；真实 models.json 不入库）
├── pyproject.toml · pytest.ini
└── （本地不入库）models.json · .venv/ · bootstrap_runs/ · v8/data/ · v9/data/ · .workbuddy/ · browser-screenshots/
```

## 6. 测试与自检

```bash
# Python 框架：324 条（根 19 + v8 285 + v9 20），其中 4 条 live 需真实 pi
python -m pytest -q                  # 全量
python -m pytest -q -m "not live"    # 320 条，不碰模型/网络
(cd v8 && python -m pytest tests/ -q)   # 只跑 v8
python -m pytest v9/tests -q            # 只跑 v9

# taskpad：Node 冒烟 + 真实 worker 链路 + 数据迁移
cd taskpad
node test/smoke.js          # 服务端 12 项
bash test/e2e-worker.sh     # 起临时 server + 真跑 worker.py，验证上下游注入与门禁
node test/migrate-check.js  # 拿 store.json 副本验证一次性迁移
```

> `test/dnd-check.js`、`test/nav-check.js` 需要本机 Edge + CDP，属于可选 UI 回归。
> `.venv/` 是第三方依赖测试环境，不属于本项目。

## 7. 配置与密钥

- 根 [models.json.example](models.json.example) 是模板；真实 `models.json` **已被 .gitignore 排除，从未入库**
- [scripts/cf/models.json](scripts/cf/models.json) 是免 Key 网关配置，显式保留在版本库
- 忽略规则集中在 [.gitignore](.gitignore)；各层另有自己的 `.gitignore`（如 [taskpad/.gitignore](taskpad/.gitignore)）

## 8. 本地数据与清理

工作区里有些**不入库但可再生成**的本地数据，总计约 **155MB**（本次整理一条没删）：

| 目标 | 内容 | 约 |
|---|---|---|
| `bootstrap_runs/` | v6 自举历史产物（留档） | 85M |
| `v8/data/` | dag8 SQLite 运行轨迹 | 33M |
| `.venv/` | 可重建的测试环境 | 32M |
| `browser-screenshots/` | 本地截图缓存 | 4.6M |
| 根目录 6 个一次性脚本 | `.fix_example.py` 等模型上限探测脚本与快照 | 28K |
| `taskpad/data/` 日志与备份 | 保留 `store.json` 和 `artifacts/` | 216K |

用 [scripts/clean-local.sh](scripts/clean-local.sh) 处理（**默认只列不删**）：

```bash
scripts/clean-local.sh                  # 列出目标与体积，不动文件
scripts/clean-local.sh safe --yes       # 只清 __pycache__ / .pytest_cache / *.egg-info
scripts/clean-local.sh bootstrap v8data --yes   # 按名字清指定目标
scripts/clean-local.sh all --yes        # 清全部（慎用）
```

脚本永远不会碰 `models.json`、`.workbuddy/`、`taskpad/data/store.json`、`taskpad/data/artifacts/`、`reports/`、`.git/`。

## 9. 本次整理未动、需你拍板的

这些属于结构决策，不在「安全整理」范围内，先记录在这里：

1. `v4/` + `v6/` 共 337 个文件仍在 main；远端已有 `vertex-edge-agent-v4`、`vea-v6-bootstrap-line` 等专门分支，可考虑移入 `archive/` 或从 main 移除。
2. `dag/` 是 v7 旧版的全量拷贝（8 个文件全不同），可改成只 re-export 的薄兼容层。
3. [pyproject.toml](pyproject.toml) 打包的包名是 `dag`（旧入口），而 canonical 是 `v7/`。
4. `.workbuddy/memory/` 是项目决策日志（含 models 上限实测、密钥泄露处置记录），目前被忽略、未入库。
5. `reports/` 已提交约 2.2MB 生成物（含 1.2MB JSON）。