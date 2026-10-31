# 开发指南（Developer Guide）

> 面向**要改代码的人**：环境、开发循环、分层规则、各模块怎么改、测试与提交门禁、真实踩坑记录。
> 仓库地图见 [../README.md](../README.md)；框架使用方式见 [../GUIDE.md](../GUIDE.md)；架构决策见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 0. 先判断你要改哪块

| 你要改的东西 | 代码位置 | 主测试 |
|---|---|---|
| 节点图 Web 应用（当前主线） | [../taskpad/](../taskpad/) · `src/{core,domain,http}/` · `public/app.js` · `worker.py` | `node test/smoke.js` · `bash test/e2e-worker.sh` |
| DAG 本体（标签匹配 / 单图 step） | [../v7/](../v7/) `engine.py` `schema.py` `matcher.py` `default_processor.py` `meta.py` | `python -m pytest tests/ -q` |
| 执行/组织层（JSON 配置图、并行、契约） | [../v8/dag8/](../v8/dag8/) · [../v8/configs/](../v8/configs/) | `(cd v8 && python -m pytest tests/ -q)` |
| 知识推理层（实体/关系/来源） | [../v9/](../v9/) `knowledge.py` `infer.py` `cycle.py` `expand.py` | `python -m pytest v9/tests -q` |
| 业务脚本（S1 / CF） | [../scripts/s1/](../scripts/s1/) · [../scripts/cf/](../scripts/cf/) | 手工跑，产物不入库 |
| 历史版本（只读参考，别在这里改） | [../v4/](../v4/) · [../v6/](../v6/) · [../dag/](../dag/) | 不在 `pytest.ini` 的 testpaths 里 |

## 1. 环境

**Python 侧**（v7/v8/v9/scripts）

- Python **3.10+**（`pyproject.toml` 要求）；依赖：`requests`、`httpx`，测试用 `pytest`（见 `pyproject.toml` 的 `[project.optional-dependencies].dev`）。
- 在仓库根直接跑测试，**不需要 pip 安装本仓库**（`pytest.ini` 已把 `tests v8/tests v9/tests` 设为 testpaths）。
- `python -m dag8` 必须在 `v8/` 目录下执行（`dag8` 包只在那里可导入），或 `PYTHONPATH=v8`。
- 模型调用：涉及 LLM 的 v7/scripts 需要根目录 `models.json`（复制 `models.json.example` 填 Key）；**离线测试不需要**。
- v8 的 `pi` 节点 / live 测试需要本机有 `pi` 二进制，先自检：

```bash
cd v8 && python -m dag8 doctor     # pi / provider 注册表 / 网关
```

**Node 侧**（taskpad）

- **Node ≥ 18，零依赖**，克隆后不需要 `npm install`。
- 可选：本机 Edge —— 只有 `taskpad/test/dnd-check.js`、`nav-check.js`（CDP/UI 回归）需要。

**Windows / WSL 注意**

- 命令里的解释器名不一致：文档示例用 `python3`（WSL），Windows 上换成 `python`。
- taskpad 执行任务命令用 `config.js` 里的 `python` 默认值；可用环境变量 `TASKPAD_PYTHON` 覆盖。
- taskpad 的种子任务命令按 `python3` 写，纯 Windows 环境需要改任务命令。

## 2. 开发循环（命令速查）

```bash
# 改完先跑离线门禁（约 20s，不需要模型/网络）
python -m pytest -q -m "not live"        # 320 条
(cd v8 && python -m pytest tests/ -q)    # v8 单跑（285 条）
python -m pytest v9/tests -q             # v9 单跑（20 条）

# taskpad：服务端冒烟 + 真实拉取链路 + 数据迁移
cd taskpad && node test/smoke.js         # 12 项，秒级
bash test/e2e-worker.sh                  # 临时 server + 真跑 worker.py
node test/migrate-check.js               # 拿 store.json 副本验迁移

# 收尾：清掉构建缓存（默认 dry-run，加了 --yes 才删）
scripts/clean-local.sh safe --yes
```

规则：

- `live` 用例（4 条，`pytest -m live`）**没有 pi 会直接 fail，不会 skip**——这是刻意设计，别改成 skip。
- taskpad 测试必须用**临时数据目录 + 独立端口**（`TASKPAD_DATA_DIR`、`SMOKE_PORT`，smoke=8411、e2e=8412），**不要**对 `taskpad/data/store.json` 写测试数据。
- v8 测试用临时 `--db` / `tmp_path`，别动 `v8/data/`。
- 前端没有构建步骤：改 `public/` 直接刷新页面（静态资源已禁缓存）；`app.js` 单文件约 2000 行，按功能区域顺序阅读。

## 3. 分层与依赖规则（必须遵守）

**Python 框架**

```
v9  →  v8  →  v7        允许的依赖方向
✗    ✗      不反向 import
```

- v7 不知道 v8/v9 的存在；v8 的 `graph` 节点通过 `v7.load` 适配；v9 复用 v8 调度。
- 单图推进只有 `v7.step()`；v8 只做"整图调度"，别再引入第二套 step API。
- `dag/` 是旧版全量拷贝，仅供老 scripts/tests 兼容；**新代码一律 `import v7`**。

**taskpad**

```
http/routes/*  →  domain/*  →  core/*        严格单向
```

- `core/` 不认识 domain，`domain/` 不认识 HTTP；`server.js` 只做装配（配置 → store → 总线 → 领域服务 → 路由 → listen）。
- 反向通信一律走进程内事件总线（`core/events.js`），不要在 domain 里直接碰 `res`/`EventSource`。
- 任何改数据的领域方法要发 `state:dirty`（SSE 侧 50ms 合并后全量广播）。

**层语义（别回退成旧模型）**

- **层属于图，不属于节点**：`graph.layer ∈ v7|v8|v9|other|null`，子图继承父图，节点继承图。
- 节点的 `engine` 只回答"这个模块靠什么被执行"：`dag8` / `v9` / `v7` / `shell` / `none`。
- 没有命令的纯架构模块用 `engine: none`，**不要伪造 `echo` 占位命令**（历史包袱，已清理）。

## 4. 各块开发要点

### 4.1 v7（DAG 本体）

- 读顺序：`schema.py`（数据结构）→ `engine.py`（step/边执行）→ `matcher.py`（tag 匹配）→ `default_processor.py` → `meta.py`（元图）。
- 模式：`load()` → `step()` → `contents`；外部算完用 `v7.set(tag, value)` 写回再继续。
- **tag 是契约**：fn 边只认精确 tag，LLM 边才做 N-gram/向量模糊；不要用近似名（`report` vs `error_report`）。
- 新增能力先看 `tests/test_v7_core.py`、`test_engine_fn.py`、`test_matcher.py` 三个用例的组织方式。

### 4.2 v8 / dag8（执行组织层）

- 模块：`config.py`（JSON → dataclass + 校验）、`scheduler.py`（按 layer 取输入、波次并行）、`nodes.py`（节点执行体）、`contracts.py`（`expect` 校验）、`store.py`（SQLite）、`pi.py`（模型后端）、`__main__.py`（CLI）。
- **6 种节点 kind**：`pi`（默认）、`graph`（v7 图；`graph.engine="v8"` 时递归跑 v8 子图）、`db`、`http`、`script`、`command`。
- 新增节点 kind 的最小步骤：
  1. `nodes.py` 写 `*Node` 类（实现 `run`），并在 `build_node()` 分发里注册；
  2. `config.py` 的 `EdgeConfig` 补字段与校验；
  3. `nodes.py` 顶部文档表 + `GUIDE.md` 同步；
  4. `v8/tests/` 补离线用例（不调模型）。
- 契约 `expect` 只挡格式：`type`(json/text)、`required_keys`、`pattern`、`schema`(子集)、`min/max_chars`；**当前策略不设字数硬上限**。
- **不用缓存**（`use_cache=False`、默认不加 `--cache`）；慢调用用 `--workers` + `--rpm` 控，不要靠加并发硬顶。
- CLI 18 个子命令：`run show runs explain diff route models stats convert cost demo abort prune cache lint doctor new scaffold`；新增子命令在 `__main__.py` 注册。
- 调试优先用：`dag8 lint`（不可达输入/环/子图路径）、`run --dry-run`（不调 pi）、`explain`（失败原因）、`stats/diff`（历史对比）。

### 4.3 v9（知识推理层）

- 模块：`knowledge.py`（entities/claims/sources/explored）、`infer.py`（选实体对 → 抓证据 → 生成 → 校验 → 入库）、`cycle.py`（一个周期 = 一张 v8 DAG）、`expand.py`（实体扩展）。
- **硬约束**：每条 claim 必须有 `source_url`（外部证据，且必须在实际抓取集合里）或 `derived_from`（推理链）；证据不足就 `none`，不许硬凑。
- 不缓存、不回灌已有 claims/最终结果；来源做归一化（Wikipedia 人类页面 → 实际抓取的 API URL）。
- 改推理路径：优先看 `configs/cycle.json`、`configs/derive.json`；离线测试在 `v9/tests/` 用 mock。

### 4.4 taskpad（服务端）

- 文件职责见 [../taskpad/README.md](../taskpad/README.md) 的「服务端结构」；核心约定：
  - 新增接口：`src/http/routes/*.js` 声明路由 → 调 domain → 返回 JSON；业务错误抛 `AppError`（`core/errors.js` 是 HTTP 状态码的唯一映射）。
  - 新增领域行为：写 `domain/*-service.js`，发事件，补 `test/smoke.js` 用例。
  - 魔法数字/上限集中在 `core/config.js` 的 `LIMITS`，不要散落。
  - **改 store 数据模型必须加迁移**（`core/store.js` 的一次性 migrate）+ 跑 `test/migrate-check.js`，否则老 `store.json` 起不来。
  - 前端：`public/index.html` + `style.css` + `app.js`；状态走 `/api/state` + SSE（`http/sse.js`），日志 SSE 与轮询双保险。
- **worker 协议**：`/api/runs/claim` → `heartbeat` → `submit` → `files`。改协议要同步 `worker.py` + smoke + e2e。
- 门禁语义：输入/输出/接口测试任一失败 → 运行 `error`，失败成果不投递下游；下游输入门禁不过 → 不执行主命令。
- UI 回归（可选）：`test/dnd-check.js`（拖拽落点）、`test/nav-check.js`（子图面包屑）需要本机 Edge 通过 CDP 连接。
- 截图放 `taskpad/docs/`，命名统一 `screenshot-*.png`，**提交前查重**（曾有 2 组完全相同的图）。

### 4.5 scripts

- 产物路径已被 gitignore（`scripts/s1/out/`、`scripts/cf/cf_report.md`），不要把结果提交。
- 新脚本按域放 `scripts/<domain>/`，不要塞进 `v7/` 或 `dag/`。
- `scripts/s1/` 需要 `models.json` + 网络；`scripts/cf/` 用 `scripts/cf/models.json` 的免 Key 网关。

## 5. 编码与提交约定

- 注释、文档、提交信息用中文；代码标识符用英文。
- 提交格式：`<type>(<scope>): <说明>`，type ∈ `feat / fix / refactor / docs / chore / test`，scope 如 `taskpad`、`v8`、`v9`。
- **一个提交一个主题**；改语义而不兼容时在 type 后加 `!`（如 `refactor(taskpad)!: ...`）。
- 行为变更要同步文档：`README.md` 测试数、`GUIDE.md`、`docs/ARCHITECTURE.md`、`v8/STATUS.md` 中至少一处会过期。
- 不提交：`models.json`（真实 Key）、`v8/data/`、`v9/data/`、`taskpad/data/`、`bootstrap_runs/`、日志、SQLite、本地截图。

## 6. 提交前检查清单

- [ ] `python -m pytest -q -m "not live"` 全绿（320 条）
- [ ] 动过 taskpad：`node test/smoke.js` 12/12，必要时 `bash test/e2e-worker.sh`
- [ ] 动过 store 结构：`node test/migrate-check.js`
- [ ] `git diff` 自查，没有密钥 / 运行数据 / 缓存进 diff
- [ ] 文档链接无断链（相对路径改过就检查）
- [ ] `scripts/clean-local.sh safe --yes` 清掉 pycache 等构建缓存
- [ ] 工作区只剩本次有意的改动

## 7. 真实踩坑记录

1. **`.gitignore` 不支持行内注释**：`bootstrap_runs/   # 注释` 会让整行失效，`#` 必须单独一行。
2. **测试会生成 `__pycache__` / `.pytest_cache`**：收尾用 `scripts/clean-local.sh safe --yes`，别再手工 `find -delete`。
3. **端口冲突**：taskpad 默认 8330，测试固定 8411/8412；被占用时先查残留进程。
4. **`python -m dag8` 找不到包**：必须在 `v8/` 下跑，或设 `PYTHONPATH=v8`。
5. **live 测试"必挂"是设计**：没有 `pi` 就 fail（不 skip），别为了本地全绿把 live 改成 skip。
6. **改 taskpad 数据模型忘了迁移**：老 `store.json` 直接起不来；迁移必须幂等 + 用副本验证。
7. **前端改了没效果**：静态资源已禁缓存，但仍在旧标签页时需要硬刷新；SSE 断开会退化为轮询，排查时看 `/api/state`。
8. **截图/README 数字容易过期**：删图前先 `grep` 引用；测试数量变了顺手改 `README.md` 第 6 节。
9. **`dag/` 不是 v7**：8 个文件全不同，改错地方会"改了没生效"。