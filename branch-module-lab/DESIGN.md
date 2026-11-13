# Branch → Module 方案探索（Go / JavaScript / Python）

> 目标：探索一套能落地的机制，满足三件事：
>
> 1. **各个 branch 实现单一功能**；
> 2. **merge 之后作为模块实现功能**；
> 3. **单一功能 branch 更新之后，merge 的模块也会更新**。
>
> 本文给出机制、对比方案、三语言实现细节与坑、存量分支迁移步骤；配套可运行实验见本仓库
> `README.md` 与 `demo/demo.sh`。

---

## 0. 结论速览（TL;DR）

| 问题 | 结论 |
|---|---|
| 用哪个 git 操作？ | **普通 `git merge`** 就够：分支只改自己的模块目录，反复 merge 即模块更新。**不需要 submodule**。 |
| 怎么让「merge 之后是模块」？ | main 上模块 = `modules/<lang>/<name>/` + `module.json` 契约 + 宿主自动发现加载（Go plugin / JS ESM / Python importlib）。 |
| 怎么让「分支更新 → 模块更新」？ | 一个**幂等同步器**（`sync.py`）循环执行 `fetch → merge --no-ff → build → lock`，由 CI（push 触发 + 定时兜底）或常驻 `watch` 执行。 |
| 存量乱分支怎么办？ | 用 `git subtree add/pull --prefix=modules/<lang>/<name>` 把分支**根目录**并成模块；或从 main 切干净分支只搬该功能文件。 |
| 三语言差异？ | JS / Python 源码直接热加载；Go plugin 有工具链/平台限制且**不可卸载**，demo 用 `re-exec` 自更新，生产建议走 RPC 子进程（hashicorp/go-plugin）或独立进程。 |
| 模块粒度？ | 本文 demo 演示「进程内函数模块」；真实整应用模块把 `module.json` 扩展出 `build/run/healthcheck`，宿主改成 supervisor，**同步机制完全一样**。 |

---

## 1. 需求拆解：三句话 = 三个机制

| 诉求 | 缺了它会怎样 | 对应机制 |
|---|---|---|
| 各个 branch 实现单一功能 | 分支之间互相踩文件，merge 冲突不断，永远合不完 | **路径契约**：一个分支只新增/修改 `modules/<lang>/<name>/`（+ 自己目录内的测试） |
| merge 之后作为模块实现功能 | merge 进来只是一堆代码，主程序不知道怎么用 | **模块契约 + 动态加载**：`module.json` 描述模块，宿主扫描目录、按语言原生机制加载、调用统一接口 |
| branch 更新后，merge 的模块也更新 | 只合一次，之后分支又往前走了，main 还是旧的 | **幂等同步循环**：反复 `merge`（不是一次性 merge），配合 lock 记录分支 commit |

第二条和第三条常被混为一谈。关键认知：

> **merge 是一次动作；「模块跟着分支更新」是持续动作。**
> 前者靠路径隔离保证不冲突，后者靠一个能反复执行、可自动化、幂等的同步器。

---

## 2. 总体架构

```
 作者 clone                      bare origin                        集成分支 main
────────────                    ────────────                        ─────────────
 feat/greet-go ──push──▶        feat/greet-go ──┐
 feat/greet-js ──push──▶        feat/greet-js ──┼── sync.py（CI / watch）
 feat/greet-py ──push──▶        feat/greet-py ──┘   fetch
                                                    merge --no-ff <每个分支>
                                                    build（Go → module.so）
                                                    write modules.lock.json
                                                    commit
                                                          │
                                                          ▼
                                     modules/go/greet-go/       modules/js/greet-js/      modules/python/greet-py/
                                     modules.lock.json（commit/version/artifact）
                                                          │
                             ┌────────────────────────────┼────────────────────────────┐
                             ▼                            ▼                            ▼
                       Go 宿主(.so)                JS 宿主(ESM)                Python 宿主(importlib)
                       加载并调用统一接口           加载并调用统一接口            加载并调用统一接口
```

三个角色：

- **作者**：只在一个 feature 分支里开发一个模块，push；
- **同步器**：唯一往 main 合并的人（CI 机器人 / 常驻进程 / 本地脚本），保证 main 是模块的「最新集成版」；
- **宿主**：只认 `modules/` 目录和契约，不关心 git。

---

## 3. 推荐方案：branch-per-module + 普通 merge + 同步器

### 3.1 模块契约（三语言统一）

目录：

```
modules/
├── go/<name>/        # module.json + *.go（package main，导出 New() sdk.Module）
├── js/<name>/        # module.json + index.mjs（export default {name, describe, run}）
└── python/<name>/    # module.json + impl.py（NAME / describe() / run(payload)）
```

`module.json`（跨语言统一，便于同步器/宿主统一处理）：

```json
{
  "name": "greet-go",
  "language": "go",
  "interface": "bml/module-v1",
  "version": "1.0.0",
  "entry": "module.so"
}
```

设计要点：

- **`interface` 是契约版本号**。宿主只加载自己认识的 interface；模块升级接口走 `module-v2`，老宿主忽略，避免一次全量升级。
- **`version` 是模块语义版本**（作者维护）；**lock 里的 commit** 才是可审计的真实来源。
- **模块名 = 目录名**，且全仓库唯一（三语言也不允许重名，否则 lock/日志会混淆）。
- 模块代码**不允许反向依赖宿主内部包**，只依赖宿主提供的 `sdk`（Go）或注入的接口对象（JS/Python）。

### 3.2 分支契约（单一功能的关键）

一个「单一功能分支」必须满足：

1. **从 `origin/main` 切出**（保持 merge 干净）；
2. **只新增/修改自己模块目录** `modules/<lang>/<name>/`（以及 `modules.json` 里自己那一行登记，见下）；
3. 分支名 `feat/<module>`（同步器按 `modules.json` 的登记找分支，命名只是可读性）；
4. 不修改其它模块、不动宿主、不动 lock（lock 是生成物）；
5. 模块内部测试放在自己目录（`*_test.go` / `test_*.py` / `*.test.mjs`），同步后统一跑。

> 为什么「只改自己目录」是硬约束？
> git 的 merge 是**按路径三方合并**。不同分支改不同目录，merge 永远不冲突；
> 一旦两个分支都改 `modules.json` 或宿主，就会天天冲突，那就退回「人肉合分支」的老路。
> `modules.json` 的登记可以由同步器从分支**约定自动推导**（分支名 → 模块名），或集中在一个「只增不改」的登记文件里用 union merge driver 处理。

### 3.3 同步循环（分支更新 → 模块更新）

`sync.py once` 的算法（**幂等**，跑 1 次和跑 N 次结果一致）：

```
1. 断言在 main、已跟踪文件干净
2. git fetch <remote> --prune
3. for module in modules.json:
     git merge --no-ff --no-edit -m "merge module <name> from <branch>" <remote>/<branch>
     # 若已在 main 中 → "Already up to date"，无副作用
     # 若冲突 → merge --abort 并报错（正常契约下不会发生）
4. for go module: go build -buildmode=plugin -o module.so.tmp && mv（原子替换）
5. 刷新 modules.lock.json（name/language/branch/commit/version/artifact）
6. 有变化则提交 lock
```

触发方式（任选/叠加）：

| 方式 | 延迟 | 说明 |
|---|---|---|
| CI：`on: push: branches: feat/**` | 秒级 | 主推荐，见 `ci/sync-modules.yml` |
| CI：定时 cron 兜底 | 分钟级 | 防止漏事件/分支保护挡住 push |
| 常驻 `sync.py watch` | 秒级 | 本地/内网集成机；demo 用这个 |
| pre-push hook（作者侧） | 即时 | 只能本地验证，不能替代 CI |

`sync.py watch` 只做一件事：比较 `modules.lock.json` 里的 commit 与 `origin/<branch>` 的 HEAD，不同就触发一次 `once`。**不要**用文件 watcher 监听 `modules/`——那只能看到 merge 结果，看不到「分支有没有新提交」。

### 3.4 冲突、回滚、版本钉住

- **冲突**：路径隔离后几乎为零；剩下的是 `modules.json` 登记行。处理：登记集中维护（同步器自动推导）或用 `.gitattributes` 的 union merge driver；实在冲突时 `sync.py` 会 abort 并报错，main 保持干净。
- **回滚**：`git revert <merge commit>` 让 main 回到旧模块；但注意**下一次 sync 会把分支最新提交再合回来**。要长期钉住版本，用「版本分支/tag」当来源：`modules.json` 里把 `branch` 换成 `ref: "release/foo@1.2"` 之类（同步器读 `ref` 字段，实现只需几行）。
- **lock 是审计与对账的锚点**：它记录每个模块对应的 commit 和产物路径。可以再加 `sha256` 校验产物、记录 sync 时间/触发者。
- **main 保护**：CI 机器人直接 push main 可能被分支保护挡住。两种做法：
  1. 用独立的 `integration` 分支做同步，再按需快进 main；
  2. 给机器人开 bypass 权限（审计 + 只允许 CI 提交）。
  推荐 1，风险最低。

### 3.5 分支落后怎么办？

- 因为路径隔离，分支**不需要**频繁 rebase / merge main；
- 但长期落后的分支在「改了共享文件」时才会冲突。可选增强：同步器 merge 成功后，反向把 main 合回分支（`git merge main` on branch），让分支始终可 merge；
- 别在分支上 rebase 已推送的公共历史——会破坏「merge 后模块追踪分支 commit」的可审计性。

---

## 4. 备选方案对比

### 4.1 git subtree（存量分支适配首选）

适合：分支**根目录就是你要的模块内容**（或某个子目录），不想重建分支历史。

```bash
# 首次：把分支根目录并成模块目录
git subtree add --prefix=modules/go/legacy origin legacy/feature

# 之后：分支更新 → 反复 pull 即可（等价于本方案的 merge）
GIT_EDITOR=true git subtree pull --prefix=modules/go/legacy origin legacy/feature

# 只取分支里的某个子目录：先 subtree split 出临时分支，再 add
git subtree split --prefix=tools/foo origin legacy/feature -b tmp/foo
git subtree add --prefix=modules/go/foo tmp/foo
```

注意点：

- `subtree pull` 的参数是 **`<repository> <ref>` 两个位置参数**，少了 ref 会报
  `fatal: you must provide <repository> <ref>`；
- 非交互环境用 `GIT_EDITOR=true` 或 `-m "..."`；
- subtree 会在 main 产生 merge commit（和普通 merge 一样可审计）；
- 分支里如果混着多个功能目录，先 split 或新建干净模块分支，别整棵树并进来。

**普通 merge vs subtree**：普通 merge 要求分支就是「main + 自己模块目录」；subtree 允许分支是任意布局，代价是 prefix 映射 + 每次 pull 多一层 split 逻辑。**新流程用普通 merge，存量分支用 subtree**。

### 4.2 git submodule

模块放独立仓库，main 用 submodule 钉住 commit。

- 优点：模块独立发布、强版本隔离、权限分离；
- 缺点：**更新不会自动来**——上游更新后要手动 `git submodule update --remote && commit`；CI 还得处理 submodule 凭据；对「分支更新自动进模块」这个诉求是**反方向**的。
- 适用：模块本身有独立生命周期/外部团队维护。本诉求不建议。

### 4.3 CI 构建产物 + registry（artifact 分发）

分支 push → CI 构建出二进制/包 → 发布到 registry（npm/PyPI/Go module proxy/OCI）→ main 只更新 lock。

- 优点：构建环境统一、模块可跨项目复用、产物不可变可签名；
- 缺点：多一套 registry/版本发布流程；「分支上的源码改动」到「可加载产物」之间多一跳，调试链路长。
- 适用：模块要被**多个宿主/项目**消费，或需要二进制分发。单仓多模块（本仓库场景）用普通 merge 更轻。

### 4.4 运行时直接读分支（不推荐）

宿主自己 `git checkout <branch>` 到 cache 目录再加载，完全不 merge。

- 能实现「分支更新模块更新」，但：
  - 构建结果不可审计（同一分支不同时间加载结果可能不同）；
  - 无法 code review / 测试门禁；
  - 版本回滚、依赖锁定都很难；
  - 生产环境让进程拉 git 是安全/稳定性风险。
- 只适合本地快速原型。

### 4.5 对比表

| 方案 | 分支更新如何进入模块 | 冲突面 | 复杂度 | 适用 |
|---|---|---|---|---|
| **普通 merge + 同步器（推荐）** | 反复 `git merge` | 极小（路径隔离） | ★ | 单仓多模块、模块=进程内扩展 |
| git subtree | 反复 `git subtree pull` | 小 | ★★ | 存量分支、分支根=模块 |
| git submodule | 手动 bump commit | 无 | ★★ | 模块独立仓库/外部团队 |
| CI artifact + registry | 发布新版本 + 更新 lock | 无 | ★★★★ | 跨项目复用、二进制分发 |
| 运行时直接读 branch | 宿主自行 checkout | 无 | ★★ | 仅本地原型，不推荐 |

---

## 5. 三语言运行时：机制与坑

### 5.1 Go：标准库 `plugin`

> 先澄清：Go **没有官方「插件框架/接口」**。`plugin` 包只提供最底层的符号查找
> （`plugin.Open` + `plugin.Lookup`）；所谓「插件接口」是我们在共享包 `go/sdk` 里
> 自己定义的 Go `interface`。产物是 Linux/macOS 的 `.so`，**不是 Windows 的 `.dll`**——
> Go 的 `-buildmode=plugin` 不支持 Windows。若在 Windows 上确实要 DLL 式扩展，只有两条路：
> `-buildmode=c-shared` + C ABI（丢掉 Go interface 的便利），或者走 5.1 末尾的 RPC 子进程（推荐）。

Go 官方的动态加载只有 `plugin`（`-buildmode=plugin`，产物 `.so`）。

宿主：

```go
p, err := plugin.Open("modules/go/greet-go/module.so")
sym, err := p.Lookup("New")                 // 插件里：func New() sdk.Module
mod := sym.(func() sdk.Module)()
mod.Run("world")
```

插件：

```go
package main

import "example.com/bml/go/sdk"

type mod struct{}
func (mod) Name() string { return "greet-go" }
func (mod) Describe() string { return "..." }
func (mod) Run(p string) (string, error) { return "hello, " + p, nil }
func New() sdk.Module { return mod{} }      // 必须导出
```

坑 / 限制（务必知道）：

1. **同一工具链**：宿主和插件必须用**同一 Go 版本 + 同一依赖版本**构建，否则 `plugin.Open` 报版本不匹配。同一仓库同一次构建天然满足。
2. **平台**：官方支持 linux / darwin / freebsd，**Windows 不支持**。
3. **类型身份**：接口/结构体必须来自**共享包**（本文的 `go/sdk`），宿主和插件都 import 它，类型断言才能成立；不要在插件里另定义一份同形状接口。
4. **不可卸载/重载**：插件一旦 `Open` 就常驻进程，没有 `Close`。更新只能**换进程**。demo 的做法是宿主检测 `module.so` mtime 变化后 `syscall.Exec` 自重启；生产更推荐第 6 节的 RPC 子进程。
5. **构建产物原子替换**：同步器先写 `module.so.tmp` 再 `rename`，避免正在运行的宿主 mmap 的旧文件被截断（直接覆盖可能 SIGBUS）。
6. **CGO / build tag**：插件构建与宿主必须一致（CGO 开关、build tags）。
7. **包初始化顺序**：插件 init 在 `Open` 时执行一次，不要依赖宿主全局状态。

**替代（生产推荐）**：`hashicorp/go-plugin`（gRPC 进程插件）或自己写 JSON-RPC over stdio。优点：进程隔离、崩溃不影响宿主、可热换、可跨语言；代价是多一个子进程和协议层。本实验的同步机制（git 层）完全复用，只是「加载」从 `.so` 换成「启动子进程」。

### 5.2 JavaScript：ESM dynamic import

> 先澄清：Node **不需要把模块编译成 DLL/插件**。`require()`（CJS）和 `import()`（ESM）
> 本身就是运行时模块加载器：每个 `.js/.mjs` 文件就是可插拔模块，「插件接口」= 约定的
> `exports` 形状（可用 TypeScript `interface` / JSON Schema 做契约校验）。
> 若要二进制插件，Node 有原生扩展 `.node`（N-API，`process.dlopen`），但那不是本方案需要的。

宿主：

```js
const st = fs.statSync(entry);
const ns = await import(pathToFileURL(entry).href + `?v=${st.mtimeMs}`);
const impl = ns.default;          // export default { name, describe, run }
await impl.run(payload);
```

坑 / 注意：

1. **ESM 有模块缓存**：同一 URL 只会执行一次（Node 的 ESM ModuleMap 按 resolved URL 缓存）。热更新必须换 URL——本项目用 `?v=mtime` 做 cache busting（Node 支持 file URL 带 query）。实测：同一 URL 第二次 `import()` 不重新执行模块；加 `?v=1` 后模块顶层代码会再执行一次。CJS 则是 `require.cache`，重载要 `delete require.cache[require.resolve(entry)]`。
2. **旧实例不会自动释放**：反复换 URL 会不断产生新模块实例，内存缓慢增长。频繁热更新/长驻进程建议用 `Worker` 隔离模块、或直接重启进程。
3. **模块依赖**：模块 `import` 的第三方包必须在宿主可解析范围内。单仓用 npm workspaces / 共享 `node_modules`；独立发布的模块自带依赖清单。
4. **安全**：动态加载 = 执行任意代码，模块来源必须受信（PR review + CI + 锁版本）。
5. **CJS 项目**：如果宿主是 CommonJS，用 `delete require.cache[require.resolve(p)]` 实现重载；新项目建议 ESM。

### 5.3 Python：`importlib` 按文件加载

宿主：

```python
spec = importlib.util.spec_from_file_location(f"bml_module_{name}", entry)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module        # 让模块内部 import 正常工作
spec.loader.exec_module(module)
module.run(payload)
```

坑 / 注意：

1. **不要用 `import modules.python.greet_py`**：`modules` 是普通目录会被当成 namespace package，且模块名冲突时会加载错。按文件路径 + 唯一模块名加载最稳。
2. **重载语义**：`importlib.reload` 不会重跑「新建的对象」，且旧引用仍指向旧模块对象。热更新建议**重新 `spec_from_file_location` + 新模块名**（本 demo 做法），或者干脆换进程（worker）。
3. **包内相对 import**：模块有自己的子包时，需要设置 `submodule_search_locations` 或把模块目录加入 `sys.path[0]`。
4. **第三方依赖**：模块 import 的库必须在宿主 venv 里；独立部署建议每个模块自己带依赖或走独立进程。
5. **隔离**：Python 无进程内沙箱，模块崩溃/死循环会拖垮宿主。生产建议每模块 worker 进程（正好也和 Go RPC 方案统一）。

### 5.4 三语言对照

| | Go | JavaScript | Python |
|---|---|---|---|
| 加载 API | `plugin.Open` | `import()` | `importlib.util.spec_from_file_location` |
| 入口符号 | `func New() sdk.Module` | `export default {...}` | `NAME/describe/run` |
| 热更新 | 不可卸载 → re-exec/RPC 子进程 | 换 URL 重新 import | 新建模块对象重载（或换进程） |
| 平台限制 | linux/darwin/freebsd | Node ≥ 18 | Python 3.8+ |
| 隔离性 | 同进程（无隔离） | 同进程（无隔离） | 同进程（无隔离） |
| 生产替代 | gRPC 子进程插件 | Worker/子进程 | worker/子进程 |
| 本实验 | `go/host/main.go` + `go/sdk` | `js/host.mjs` | `python/host.py` |

一句话：**「模块加载」三语言各有原生答案；「模块更新」统一交给 git 同步器**，两者解耦，所以同一套流程可以跨语言。

### 5.5 额外路线：C ABI 共享库（`.so` / `.dll`），一份二进制多语言共用

如果目标不是「Go 模块给 Go 宿主、JS 模块给 Node 宿主」，而是**一份二进制被 Go / Node / Python 同时加载**，那就走 C ABI 共享库：

- Linux 编 `.so`、Windows 编 `.dll`、macOS 编 `.dylib`，**代码和接口完全一样**；
- 接口只能是 C 类型（`int` / `char*` / buffer / 函数指针），不能是 Go interface 或 JS 对象；
- Go 侧：Linux/macOS 用 cgo `dlopen/dlsym`；Windows 用 `syscall.NewLazyDLL`（纯 Go，无需 cgo）；
- Node 侧：**不能直接 `require` 裸库**，必须包一层 N-API addon（`.node`，内部 `dlopen`/`LoadLibraryA`），或使用 FFI 库（koffi/ffi-napi）；
- Go 标准库 `plugin` 与这条路无关：它只能加载 Go 编译的 `.so`，且不支持 Windows。

实测（本机 Linux，详见 `abi-plugin/README.md`）：

| 指标 | Go | Node |
|---|---|---|
| 跨语言调用开销 | ~44 ns/次（cgo，百万次循环） | ~30 ns/次（N-API，百万次循环） |
| 同语言基线 | pure Go ~0.22 ns | pure JS ~1.5 ns |
| 一次调用批量算 1e6 次 | ~0.22 ns/元素 | ~0.23 ns/元素 |
| 加载耗时（一次性） | `dlopen` ~2 ms | addon+dlopen ~8 ms |

结论：**边界调用次数少、单次负载大 → 划算；高频小函数 → 合并成批量接口**。C ABI 解决「同进程跨语言」，崩溃隔离/热替换仍应上 RPC 子进程（见第 6 节）。

---

## 6. 进程级模块：真实应用的扩展

本 demo 的模块是「函数模块」（进程内调用）。真实项目里的功能分支往往是**整应用**（HTTP 服务、worker、CLI），这时：

```json
{
  "name": "taskpad",
  "language": "js",
  "interface": "bml/service-v1",
  "version": "1.0.0",
  "entry": "server.js",
  "build": ["node", "--check", "server.js"],
  "run": ["node", "server.js", "--port", "{port}"],
  "healthcheck": "http://127.0.0.1:{port}/api/health"
}
```

宿主从「动态 import」换成 **supervisor**：

- `sync.py` 合并模块后执行 `build`；
- supervisor 按 `run` 拉起进程，读 `healthcheck` 判活，崩溃重启；
- 模块更新后优雅重启对应进程（`SIGTERM` → 等待 → 启动新版本）；
- 语言只影响 `build/run` 命令，**git 同步机制一行不改**。

这与本仓库 `taskpad` 的「网站调度 + worker 拉任务」模型天然契合：模块就是可被调度的服务单元。

---

## 7. 迁移本仓库（agent 仓库）现有分支的建议

现状：`origin` 上是一批历史分支（`tampermonkey`、`openai-adapter-enhanced`、`step-by-step`、`vertex-edge-agent-v4`、`cell-agent`、`v4/v6` 备份等），每个分支是**整仓库快照**，不是「单一功能 + 干净路径」。直接原样 merge 会互相覆盖、冲突爆炸。

推荐迁移步骤：

1. **盘点**：对每个分支确定「它真正对应的功能/目录」：
   ```bash
   git log --oneline main..origin/<branch> | head
   git diff --stat main...origin/<branch>
   git ls-tree --name-only origin/<branch>
   ```
2. **固定模块清单**：确定要长期维护的模块（如 `gitproxy`(Go)、`taskpad`(JS)、`v8`(Python)…），给每个分配 `modules/<lang>/<name>/`。
3. **新建干净模块分支**（不要直接改旧分支）：
   ```bash
   git checkout -b feat/<module> origin/main
   # 从旧分支取该功能文件（保留文件内容，历史按需保留）
   git checkout origin/<old-branch> -- <tool-dir>
   # 移动到 modules/<lang>/<name>/ 并补 module.json
   git add -A && git commit -m "feat(<module>): 从 <old-branch> 迁入为模块"
   git push -u origin feat/<module>
   ```
   想保留历史：`git subtree split --prefix=<tool-dir> origin/<old-branch> -b split/<module>`，再 `git subtree add` 进模块目录。
4. **旧分支冻结为只读**，之后只在新模块分支开发；需要查旧历史时保留分支即可。
5. **同步器试运行**：先在独立 `integration` 分支跑 `sync.py once` + 全量测试，稳定后再保护 main。
6. **目录隔离**：迁移初期允许模块目录里保留自己的 `package.json` / `go.mod` / `requirements.txt`；等稳定后再统一到仓库级构建。

> 注意：本仓库是**多项目工作区**（`taskpad` / v7-v9 / gitproxy / 各脚本），不是单一应用。「模块」边界按功能目录划，不要按文件类型划。

---

## 8. 安全、测试与可观测性

- **安全**：能动态加载的模块 = 能执行任意代码。模块分支必须走 PR review + CI；同步机器人只信任 `origin` 上的受保护分支；必要时对产物签名/校验 sha256。
- **测试门禁**：同步后跑模块自带测试 + 宿主冒烟（加载全部模块、调用一次 `run`）。测试失败则不同步该分支（`sync.py` 可扩展 `--test` 钩子）。
- **可观测**：lock 文件记录 commit；宿主启动日志打印 `name/version/commit`；sync 每次变更打一行日志（谁、哪个分支、新旧 commit）。
- **回滚演练**：定期验证 `git revert` 一个模块 merge 后系统仍可启动。

---

## 9. 实验验证（本仓库）

```bash
cd branch-module-lab
bash demo/demo.sh
```

预期结论：

1. 3 个 feature 分支 push 后，`sync.py once` 让 main 出现 `modules/go|js|python/greet-*`，且 git log 有 3 个 merge commit；
2. 三语言宿主分别输出 `hello, world (go|js|python v1)`；
3. 作者 push v2 后，常驻 `sync.py watch` 自动 merge，模块源码与 lock 都更新到 v2；
4. 宿主 `--watch` 下作者 push v3：sync merge → Go 插件重建并 re-exec、JS 重新 import、Python 重新加载，三端输出 v3，全程无人工。

日志在 `demo/logs/`（`sync-v1.log` / `sync-watch.log` / `host-*-v1..v3.log` / `git-log.txt`）。

---

## 10. 落地顺序建议

1. **先选最简单的语言落地**（JS 或 Python），把 `module.json` + 宿主加载 + `sync.py once` 跑通；
2. **加 CI**：`push: feat/**` 触发 `sync.py once` + 测试；`schedule` 兜底；
3. **迁移 1~2 个真实分支**（用 subtree 或干净重建），验证路径隔离是否真的不冲突；
4. **再上 Go**：若模块是进程内扩展，用 plugin 注意平台/工具链；否则直接走 RPC 子进程方案（更稳）；
5. **最后补**：分支回同步、lock 校验、回滚演练、监控告警。

**一句话总结**：把「分支到模块」做成语义明确的**持续同步**，而不是一次性 merge；再把「模块加载」交给各语言原生机制。两层解耦后，Go / JS / Python 可以共用同一套 branch → module 流程。
