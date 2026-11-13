# branch-module-lab：branch → module 的可运行实验

> 验证一个工作流：**每个 branch 只实现单一功能；merge 之后作为模块；branch 更新后，merge 进来的模块自动跟着更新**。
> 三套语言运行时：**Go（plugin） / JavaScript（ESM dynamic import） / Python（importlib）**。
>
> 方案对比、原理、坑、存量分支迁移方法见 **[DESIGN.md](DESIGN.md)**；
> **全部知识（git 同步层 + 运行时加载层 + C ABI/DLL + 性能实测 + 依赖 + 选型）见完整手册 [HANDBOOK.md](HANDBOOK.md)**。

## 核心机制（三句话对应三个机制）

| 诉求 | 实现 |
|---|---|
| 各个 branch 实现单一功能 | 分支契约：只新增/修改 `modules/<lang>/<name>/` 一个目录，不碰别人的路径 |
| merge 之后作为模块实现功能 | main 上「模块」= 目录 + `module.json` + 宿主自动发现加载（三语言原生机制） |
| 单一功能 branch 更新后，merge 的模块也更新 | `sync.py` 幂等循环：`fetch → merge --no-ff → build → lock`，由 CI 或常驻 watch 持续执行 |

```
作者 clone                        origin.git（bare）                 branch-module-lab（main）
──────────                        ──────────────────                 ─────────────────────────
feat/greet-go  ──push──▶         feat/greet-go ──fetch/merge──▶      modules/go/greet-go/
feat/greet-js  ──push──▶         feat/greet-js ──fetch/merge──▶      modules/js/greet-js/       ──▶ Go/JS/Python 宿主
feat/greet-py  ──push──▶         feat/greet-py ──fetch/merge──▶      modules/python/greet-py/      动态加载执行
                                                                     modules.lock.json（commit/version/产物）
```

关键点：**merge 不是一次性动作**。分支每次 push，同步器都会再 merge 一次，模块目录因此一直是分支的最新内容。

## 目录结构

```
branch-module-lab/
├── modules.json           # 登记：哪个 branch 提供哪个模块（单一事实来源）
├── modules.lock.json      # 生成物：每个模块当前对应的 commit / version / artifact
├── sync.py                # 同步器：status / once / watch
├── go/
│   ├── sdk/sdk.go         # Go 模块契约（Module 接口）
│   └── host/main.go       # Go 宿主：plugin.Open 加载 modules/go/*/module.so
├── js/host.mjs            # JS 宿主：ESM dynamic import（?v=mtime 破缓存）
├── python/host.py         # Python 宿主：importlib 按文件路径加载
├── modules/<lang>/<name>/ # 由 feature 分支 merge 出来的模块（main 上不手写）
├── demo/
│   ├── author_modules.py  # 模拟模块作者：建分支 / 升级版本 / push
│   └── demo.sh            # 端到端演示（v1 → v2 → v3 全自动）
├── abi-plugin/            # 附加实验：C ABI 共享库（Linux .so / Windows .dll）被 Go/Node 共用
├── HANDBOOK.md            # 完整手册（git 同步 + 运行时加载 + C ABI + 性能 + 选型）
└── ci/sync-modules.yml    # CI 示例：push feat/** 或定时触发同步
```

## 快速开始

> demo 会重置本仓库 `main` 到 `demo-base` tag，只影响这个实验仓库。

```bash
cd branch-module-lab
bash demo/demo.sh
```

演示内容：

1. 作者在 3 个 feature 分支分别实现一个模块（Go / JS / Python），push 到 bare origin；
2. `sync.py once` 把 3 个分支 merge 成 main 上的 `modules/*`，Go 模块自动编译成 `module.so`；
3. 三语言宿主分别加载并执行模块，打印 `hello, world (go v1)` 等；
4. `sync.py watch` 常驻，作者把分支升到 v2 → 模块自动更新；
5. 三语言宿主 `--watch`，作者升到 v3 → **分支 push → 自动 merge → 运行时热更新** 全链路无需人工。

日志输出在 `demo/logs/`（gitignore）。

### 手动流程（不跑 demo）

```bash
# 构建 Go 宿主
go build -o .bin/go-host ./go/host

# 看某个模块分支的同步状态
python3 sync.py status

# 同步一次（合并所有 modules.json 登记的分支）
python3 sync.py once

# 三语言宿主各跑一次
./.bin/go-host --root .
node js/host.mjs --root .
python3 python/host.py --root .

# 常驻自动同步（CI 里换成 sync.py once 即可）
python3 sync.py watch --interval 2
```

## 新增一个模块（推荐流程）

```bash
# 1. 从 main 切出只做一件事的分支（在作者 clone 里）
git checkout -b feat/weather origin/main

# 2. 只新增自己模块目录
mkdir -p modules/js/weather
cat > modules/js/weather/module.json   # {"name":"weather","language":"js","interface":"bml/module-v1","version":"1.0.0","entry":"index.mjs"}
cat > modules/js/weather/index.mjs     # export default { name, describe, run(payload) }

# 3. 提交并 push
git add modules/js/weather && git commit -m "feat(weather): add weather module" && git push -u origin feat/weather

# 4. 在 main 的 modules.json 登记
#    {"name":"weather","language":"js","branch":"feat/weather"}

# 5. 同步 → 模块出现
python3 sync.py once
```

之后这个分支每 push 一次，`sync.py`（watch 或 CI）再把新提交 merge 进 main，`modules/js/weather/` 就更新了。

## 三种语言的模块契约

| 语言 | 模块入口 | 宿主加载方式 | 契约 |
|---|---|---|---|
| Go | `modules/go/<name>/`，`package main` 导出 `New() sdk.Module` | `plugin.Open("module.so")` → `Lookup("New")` | `sdk.Module{Name, Describe, Run}` |
| JS | `modules/js/<name>/index.mjs` 默认导出对象 | `await import(fileURL + "?v=" + mtime)` | `{ name, describe(), run(payload) }`（可 async） |
| Python | `modules/python/<name>/impl.py` | `importlib.util.spec_from_file_location` | `NAME` / `describe()` / `run(payload)` |

## 附：C ABI 共享库（.so / .dll）跨语言复用

如果你想要的是「**一份二进制，Go 和 Node 都能加载**」而不是「各自语言的模块」，那是另一条路：
C ABI 共享库（Linux `.so` / Windows `.dll` / macOS `.dylib`）+ 各语言薄包装。

- Go：Linux 用 cgo `dlopen`，Windows 用 `syscall.NewLazyDLL`（纯 Go，无需 cgo）；
- Node：不能直接 `require` 裸库，要包一层 **N-API addon（`.node`）**（或 FFI 库）；
- 性能：跨语言调用约 30~45 ns/次，批量接口摊薄后可忽略；加载 `dlopen` 一次性 2~8 ms。

完整原理、代码、实测数据见 **[abi-plugin/README.md](abi-plugin/README.md)**，一键复现：

```bash
bash abi-plugin/run.sh
```

`module.json` 三语言共用：

```json
{
  "name": "greet-go",
  "language": "go",
  "interface": "bml/module-v1",
  "version": "1.0.0",
  "entry": "module.so"
}
```

## 结论（详见 DESIGN.md）

- **普通 merge 足够**：只要分支遵守「只改自己模块目录」的契约，反复 merge 就是模块更新，不需要 submodule；
- **存量分支**（任意根布局、历史很旧）用 `git subtree add/pull --prefix=modules/<lang>/<name>` 适配；
- **更新传播**交给幂等同步器 + CI/watch，不要靠人肉 merge；
- **运行时**：JS / Python 天然支持热加载；Go plugin 不可卸载，demo 用 `re-exec` 自更新，生产建议 RPC 子进程（hashicorp/go-plugin）或独立进程；
- 本实验只演示「进程内函数模块」。真实项目里的整应用模块（server/worker）可以把 `module.json` 扩展出 `build` / `run` / `healthcheck`，宿主换成 supervisor，同步机制完全不变。
- 如需**一份二进制跨语言复用**，见 [abi-plugin/README.md](abi-plugin/README.md)：C ABI 共享库（`.so`/`.dll`）可同时被 Go（cgo / `NewLazyDLL`）和 Node（N-API addon）加载。
