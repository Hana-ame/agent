# Branch → Module 与跨语言模块加载 · 完整手册

> 本文把所有讨论收敛成一份可长期参考的文档：从 **git 层「分支怎么变成模块、怎么持续更新」**，到 **运行时层「模块怎么被加载、各方案能力与性能衰减」**，再到 **C ABI 共享库（.so / .dll）、Windows 限制、依赖关系、分发与选型**。
>
> 配套可运行实验在 [README.md](README.md)：
> - 分支→模块 + 三语言宿主：跑 `bash demo/demo.sh`
> - C ABI 共享库（Go / Node / Python）：跑 `bash abi-plugin/run.sh`
>
> 实测环境：WSL2 Linux（x86-64）、Go 1.22.2、Node 22、Python 3.12、gcc 13、MinGW-w64。**性能数字仅用于量级比较**，换机器会浮动，但数量级关系稳定。

---

## 目录

1. 问题定义与两条主线
2. 总体架构
3. git 同步层：分支如何成为「活的模块」
4. 运行时加载层：方案全景
5. 方案 A：同语言原生加载（Go plugin / Node ESM / Python importlib）
6. 方案 B：C ABI 共享库（.so / .dll / .dylib）
7. 方案 C：RPC 子进程
8. 方案 D：WASM 与编译期静态方案
9. 能力矩阵
10. 性能衰减：实测数据与解读
11. 依赖关系：菱形依赖与模块间依赖
12. 版本、锁定、回滚、可观测性
13. 安全与隔离
14. 分发与构建矩阵
15. 选型决策树
16. 本实验仓库结构与命令
17. 常见误解与坑清单
18. 附录：术语表 / 实测原始数据

---

## 1. 问题定义与两条主线

需求原文：

> 各个 branch 实现单一功能；merge 之后作为模块实现功能；单一功能 branch 更新之后，merge 的模块也会更新。
> 语言涉及 Go、JavaScript、Python。

里面其实藏着两条**互相独立**的主线：

| 主线 | 解决的问题 | 技术位置 |
|---|---|---|
| **git 同步层** | 分支 → 模块目录；分支更新 → 模块更新 | git、CI、同步器 |
| **运行时加载层** | 模块目录 → 可执行功能 | 各语言加载机制（plugin / import / FFI / RPC） |

两条主线**解耦**非常重要：

- 换语言、换平台、换加载方式，git 同步层不用改；
- 反过来，git 历史怎么组织，不影响运行时怎么加载；
- 所以：**git 层可以统一成一套，加载层按部署环境分级选择。**

最容易踩的认知错误：把「merge」当成一次性动作。**merge 是一次动作，而「模块跟随分支更新」是持续动作**，必须有一个幂等、可自动化的同步器。

---

## 2. 总体架构

```
 作者 clone                    origin（bare）                    集成分支 main
────────────                  ────────────────                  ─────────────────────────────
 feat/module-a  ──push──▶      feat/module-a ──┐
 feat/module-b  ──push──▶      feat/module-b ──┼── sync.py（CI / watch）
 feat/module-c  ──push──▶      feat/module-c ──┘     fetch
                                                     merge --no-ff <每个分支>
                                                     build（Go → module.so）
                                                     写 modules.lock.json
                                                     commit
                                                           │
                                                           ▼
                                     modules/go/<name>/      modules/js/<name>/      modules/python/<name>/
                                     modules.lock.json（branch / commit / version / artifact）
                                                           │
                              ┌────────────────────────────┼────────────────────────────┐
                              ▼                            ▼                            ▼
                       Go 宿主（plugin）            Node 宿主（ESM import）       Python 宿主（importlib）
                       加载并调用统一接口            加载并调用统一接口             加载并调用统一接口
```

三个角色：

- **作者**：只在一个 feature 分支里开发一个模块；
- **同步器**：唯一往 main 合并的人（CI 机器人 / 常驻进程 / 本地脚本）；
- **宿主**：只认 `modules/` 目录和契约，不关心 git。

如果模块需要跨语言复用同一份二进制，把「宿主 ↔ 模块」的边界从同语言调用换成 **C ABI 共享库**或 **RPC 子进程**，git 层不变（见第 6、7 章）。

---

## 3. git 同步层：分支如何成为「活的模块」

### 3.1 模块契约

目录约定：

```
modules/
├── go/<name>/        # module.json + *.go（package main，导出 New() sdk.Module）
├── js/<name>/        # module.json + index.mjs（export default { name, describe, run }）
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

字段说明：

| 字段 | 作用 | 注意 |
|---|---|---|
| `name` | 模块唯一标识，等于目录名 | 三语言也不允许重名，否则 lock/日志会混 |
| `language` | 同步器决定构建方式 | go 需编译；js/python 直接加载源码 |
| `interface` | 契约版本 | 宿主只加载认识的版本；升级走 v2，避免全量升级 |
| `version` | 模块作者维护的语义版本 | 真正可审计的是 lock 里的 commit |
| `entry` | 加载入口 | Go: `module.so`；JS: `index.mjs`；Py: `impl.py` |

设计原则：

- 模块**不允许反向依赖宿主内部包**，只依赖宿主提供的 `sdk`（Go）或注入对象（JS/Python）；
- 模块 **不要 import 另一个模块**（见第 11 章依赖问题）；
- 契约里只放跨语言都成立的东西，语言特有行为写进各自 README。

### 3.2 分支契约（「单一功能」的硬约束）

一个单一功能分支必须满足：

1. 从 `origin/main` 切出；
2. **只新增/修改 `modules/<lang>/<name>/`**（以及集中维护的登记文件，见下）；
3. 分支名 `feat/<module>`（约定，便于 CI 过滤）；
4. 不修改其它模块、不动宿主、不动 lock（lock 是生成物）；
5. 模块自带测试放在自己目录（`*_test.go` / `test_*.py` / `*.test.mjs`）。

为什么「只改自己目录」是硬约束？

> git merge 是**按路径三方合并**。不同分支改不同目录，merge 永远不冲突；
> 一旦两个分支都改 `modules.json` 或宿主，就会天天冲突，退回「人肉合分支」的老路。

`modules.json` 的登记有两种做法：

- **集中登记**（本实验）：同步器读 `modules.json` 里的 `{name, language, branch, remote}`；
- **自动推导**：分支名 `feat/<module>` → 模块名，同步器扫远端分支自动登记，避免登记文件冲突；
- 若必须让分支改登记文件，用 `.gitattributes` 的 union merge driver，或把登记拆成「每模块一个文件」。

### 3.3 同步循环（幂等算法）

`sync.py once` 的核心：

```
1. 断言在 main 分支、已跟踪文件干净
2. git fetch <remote> --prune
3. for module in modules.json:
     git merge --no-ff --no-edit -m "merge module <name> from <branch>" <remote>/<branch>
     # 已合过 → "Already up to date"，无副作用
     # 冲突   → merge --abort 并报错（正常契约下不会发生）
4. for go module:
     go build -buildmode=plugin -o module.so.tmp ./modules/go/<name>
     mv module.so.tmp module.so          # 原子替换
5. 刷新 modules.lock.json（name/language/branch/commit/version/artifact）
6. 有变化则提交 lock
7. 有变化则返回 changed，供 CI/watch 决定是否通知/重启
```

要点：

- **幂等**：跑 1 次和跑 N 次结果一致，重复 merge 无副作用；
- **分支不存在**：跳过并告警，不阻断其它模块；
- **merge 后 main 只前进**，不回退、不 rebase，保留可审计历史；
- **lock 与 merge 分开提交**：merge 提交反映源码变化，lock 提交反映集成状态。

### 3.4 冲突、回滚、版本钉住

**冲突**

- 正常契约下没有冲突（路径隔离）；
- 仅有的冲突面是登记文件、宿主文件、共享配置；
- 处理策略：登记集中维护 / union merge driver / 冲突时 sync abort 并报警，main 保持干净。

**回滚**

- `git revert <merge commit>` 让 main 回到旧模块；
- 注意：**下一次 sync 会把分支最新提交再合回来**；
- 要长期钉住版本，用「版本分支/tag」当来源：`modules.json` 里把 `branch` 换成 `ref: "release/foo@1.2"`（同步器扩展几行即可）；
- 应急时也可以让同步器临时跳过某模块（`skip` 字段）。

**main 保护**

- CI 机器人直接 push main 可能被分支保护挡住；
- 方案 1（推荐）：独立 `integration` 分支做同步，验证后再快进 main；
- 方案 2：给机器人 bypass 权限（审计 + 只允许 CI 提交）。

### 3.5 触发方式

| 方式 | 延迟 | 说明 |
|---|---|---|
| CI：`on: push: branches: feat/**` | 秒级 | 主推荐 |
| CI：定时 cron 兜底 | 分钟级 | 防漏事件/分支保护挡住 push |
| 常驻 `sync.py watch` | 秒级 | 本地/内网集成机 |
| pre-push hook（作者侧） | 即时 | 只能本地验证，不能替代 CI |

`sync.py watch` 只做一件事：比较 lock 里的 commit 与 `origin/<branch>` 的 HEAD，不同就触发一次 `once`。

> **不要**用文件 watcher 监听 `modules/` 目录——那只能看到 merge 结果，看不到「分支有没有新提交」。

### 3.6 备选方案对比

| 方案 | 分支更新如何进入模块 | 冲突面 | 复杂度 | 适用场景 |
|---|---|---|---|---|
| **普通 merge + 同步器（推荐）** | 反复 `git merge` | 极小（路径隔离） | ★ | 单仓多模块、模块=扩展点 |
| **git subtree** | 反复 `git subtree pull` | 小 | ★★ | 存量分支、分支根=模块目录 |
| **git submodule** | 手动 bump commit | 无 | ★★ | 模块独立仓库/外部团队维护 |
| **CI artifact + registry** | 发布新版本 + 更新 lock | 无 | ★★★★ | 跨项目复用、二进制分发 |
| **运行时直接读 branch** | 宿主自行 checkout 到缓存 | 无 | ★★ | 仅本地原型，不推荐 |

#### 3.6.1 git subtree（存量分支适配）

适合：分支**根目录就是你要的模块内容**，或分支某个子目录是模块。

```bash
# 首次：把分支根目录并成模块目录
git subtree add --prefix=modules/go/legacy origin legacy/feature

# 之后：分支更新 → 反复 pull（等价于本方案的 merge）
GIT_EDITOR=true git subtree pull --prefix=modules/go/legacy origin legacy/feature

# 只取分支里的某个子目录：先 split 出临时分支，再 add
git subtree split --prefix=tools/foo origin legacy/feature -b tmp/foo
git subtree add --prefix=modules/go/foo tmp/foo
```

坑：

- `subtree pull` 的位置参数是 **`<repository> <ref>` 两个**，少了 ref 会报
  `fatal: you must provide <repository> <ref>`；
- 非交互环境用 `GIT_EDITOR=true` 或 `-m "..."`；
- subtree 产生 merge commit（可审计），但每次 pull 会多一层 split 逻辑；
- 分支里混着多个功能目录时，先 split 或新建干净模块分支。

#### 3.6.2 submodule

模块独立仓库，main 用 submodule 钉 commit。

- 优点：独立发布、强版本隔离、权限分离；
- 缺点：**更新不会自动来**，上游更新后要手动 `git submodule update --remote && commit`；
- 对「分支更新自动进模块」这个诉求是反方向的，本场景不推荐。

#### 3.6.3 CI 构建产物 + registry

分支 push → CI 构建 → 发布到 registry（npm / PyPI / Go proxy / OCI）→ main 只更新 lock。

- 优点：构建环境统一、可跨项目复用、产物不可变可签名；
- 缺点：多一套版本发布流程，源码到可加载产物多一跳；
- 适用：模块要被多个宿主/项目消费，或需要二进制分发。

#### 3.6.4 运行时直接读分支（不推荐）

宿主自己 `git checkout` 到 cache 再加载：能实现更新，但构建结果不可审计、无法 code review、版本回滚困难、生产环境让进程拉 git 有安全风险。

### 3.7 存量仓库迁移步骤

针对「旧分支是整仓快照、不是单一功能」的仓库（例如本工作区上层的 agent 仓库）：

1. **盘点**每个分支真正对应的功能/目录：
   ```bash
   git log --oneline main..origin/<branch> | head
   git diff --stat main...origin/<branch>
   git ls-tree --name-only origin/<branch>
   ```
2. **确定模块清单**，给每个分配 `modules/<lang>/<name>/`；
3. **新建干净模块分支**（不要直接改旧分支）：
   ```bash
   git checkout -b feat/<module> origin/main
   git checkout origin/<old-branch> -- <tool-dir>     # 取该功能文件
   # 移动到 modules/<lang>/<name>/ 并补 module.json
   git add -A && git commit -m "feat(<module>): 从 <old-branch> 迁入为模块"
   git push -u origin feat/<module>
   ```
   想保留历史：`git subtree split --prefix=<tool-dir> origin/<old-branch> -b split/<module>`，再 subtree add；
4. **旧分支冻结只读**，之后只在新模块分支开发；
5. **先集成到 `integration` 分支**跑同步 + 全量测试，稳定后再保护 main；
6. 模块目录可保留自己的 `package.json` / `go.mod` / `requirements.txt`，稳定后再统一到仓库级构建。

---

## 4. 运行时加载层：方案全景

| 方案 | 代表实现 | 调用开销量级 | 跨语言 | Windows | 热更新 | 隔离 | 类型能力 |
|---|---|---|---|---|---|---|---|
| **同语言原生** | Go `plugin`；Node `import()`；Python `importlib` | ns 级 | ❌ | Go ❌，Node/Py ✅ | Go ❌，Node/Py ✅ | ❌ | 完整 |
| **C ABI 共享库** | `.so/.dll/.dylib` + cgo / `NewLazyDLL` / N-API / ctypes | 30~650 ns | ✅ | ✅（DLL） | 有限 | ❌ | 仅 C 类型 |
| **RPC 子进程** | JSON-RPC / gRPC / hashicorp/go-plugin | 数十 µs~数百 µs | ✅ | ✅ | ✅ | ✅ | 协议类型 |
| **WASM** | wasmtime / wasmer | 亚 µs 级 | ✅ | ✅ | ✅ | ✅ 沙箱 | 受限（host 提供） |
| **编译期静态** | 库代码 / 单体 | 0（原生调用） | ❌ | ✅ | ❌ | ❌ | 完整 |

---

## 5. 方案 A：同语言原生加载

### 5.1 Go：标准库 `plugin`

**先澄清**：Go **没有官方「插件框架/接口」**。`plugin` 包只提供最底层的符号查找（`Open` + `Lookup`）；「插件接口」是使用者在共享包里自定义的 Go `interface`。

插件：

```go
package main

import "example.com/bml/go/sdk"

type mod struct{}
func (mod) Name() string            { return "greet-go" }
func (mod) Describe() string        { return "..." }
func (mod) Run(p string) (string, error) { return "hello, " + p, nil }

// 必须导出的符号
func New() sdk.Module { return mod{} }
```

编译：

```bash
go build -buildmode=plugin -o module.so ./modules/go/greet-go
```

宿主：

```go
p, err := plugin.Open("module.so")
sym, err := p.Lookup("New")
mod := sym.(func() sdk.Module)()
mod.Run("world")
```

类型身份：共享接口必须来自**同一个共享包**（本文是 `go/sdk`），宿主与插件都 import 它。导出变量时 `Lookup` 得到的是 `*sdk.Module`，导出函数时得到 `func() sdk.Module`——后者更直观。

限制（务必知道）：

1. **平台**：官方支持 Linux / macOS / FreeBSD，**Windows 不支持**（实测 `GOOS=windows go build -buildmode=plugin` 直接报
   `-buildmode=plugin not supported on windows/amd64`）；
2. **同一工具链**：宿主和插件必须用同一 Go 版本 + 同一依赖版本 + 同样 build tags 构建，否则 `plugin.Open` 报版本不匹配；
3. **不可卸载**：插件一旦 `Open` 常驻进程，没有 `Close`；更新只能换进程。实验用「检测 `.so` 变化 → `syscall.Exec` 自重启」；
4. **构建产物原子替换**：先写 `module.so.tmp` 再 `rename`，避免正在运行的宿主 mmap 的旧文件被截断（直接覆盖可能 SIGBUS）；
5. **插件不能互相引用**：A 插件不能 `Lookup` B 插件的符号；A import B 的代码会被静态编进 A.so，产生副本、状态不共享；
6. **包初始化顺序**：插件 init 在 `Open` 时执行一次，不要依赖宿主全局状态。

生产替代：**RPC 子进程**（hashicorp/go-plugin / 自写 JSON-RPC over stdio）——进程隔离、可热换、跨平台。

### 5.2 Node：ESM dynamic import

Node **不需要编译成 DLL/插件**：`import()` / `require()` 本身就是运行时模块加载器，每个 `.mjs/.js` 文件就是可插拔模块，「插件接口」= 约定的 `exports` 形状（可用 TypeScript `interface` / JSON Schema / zod 校验）。

```js
// host.mjs
const st = fs.statSync(entry);
const ns = await import(`${pathToFileURL(entry).href}?v=${st.mtimeMs}`);
const impl = ns.default;   // export default { name, describe(), run(payload) }
await impl.run("world");
```

坑 / 注意：

1. **ESM 有模块缓存**：同一 URL 只执行一次（ModuleMap 按 resolved URL 缓存）；热更新必须换 URL，本项目用 `?v=mtime` 破缓存；
2. **旧实例不自动释放**：反复换 URL 会不断产生新模块实例；长驻高频更新建议 Worker 隔离或重启进程；
3. **CJS**：`require.cache`，重载要 `delete require.cache[require.resolve(entry)]`；新项目建议 ESM；
4. **依赖**：模块 import 的第三方包必须在宿主可解析范围内（npm workspaces / 共享 node_modules）；
5. **安全**：动态加载 = 执行任意代码，来源必须受信。

### 5.3 Python：`importlib` 按文件加载

```python
spec = importlib.util.spec_from_file_location(f"bml_module_{name}", entry)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module       # 让模块内部 import 正常工作
spec.loader.exec_module(module)
module.run(payload)
```

坑 / 注意：

1. **不要用 `import modules.python.xxx`**：`modules` 是普通目录，会被当 namespace package，容易加载错；按文件路径 + 唯一模块名最稳；
2. **重载语义**：`importlib.reload` 不会重跑「新建的对象」，旧引用仍指向旧模块对象；热更新建议重新 `spec_from_file_location` + 新模块名，或换进程；
3. **包内相对 import**：模块有子包时需要设置 `submodule_search_locations` 或调整 `sys.path`；
4. **依赖**：第三方库必须在宿主 venv；
5. **隔离**：Python 无进程内沙箱，模块死循环/崩溃会拖垮宿主；生产建议每模块 worker 进程。

### 5.4 三语言对照

| | Go | JavaScript | Python |
|---|---|---|---|
| 加载 API | `plugin.Open` | `import()` | `importlib.util.spec_from_file_location` |
| 入口符号 | `func New() sdk.Module` | `export default {...}` | `NAME/describe/run` |
| 热更新 | 不可卸载 → re-exec / RPC | 换 URL 重新 import | 新建模块对象重载（或换进程） |
| 平台 | linux/darwin/freebsd | Node ≥ 18 | Python 3.8+ |
| 生产替代 | gRPC 插件/子进程 | Worker/子进程 | worker/子进程 |

---

## 6. 方案 B：C ABI 共享库（.so / .dll / .dylib）

### 6.1 定位

一份二进制（C ABI），被 **Go / Node / Python** 等多个宿主加载。Linux `.so`、Windows `.dll`、macOS `.dylib`，**同一份源码**，接口不变。

和 Go `plugin` 的区别：

| | Go `plugin` | C ABI 共享库 |
|---|---|---|
| 产物 | Go 编译的 `.so` | 任意语言的 `.so/.dll/.dylib` |
| 加载者 | Go 宿主 | 任何支持 FFI 的宿主 |
| 接口 | Go interface（仅 Go 能用） | C 类型（跨语言） |
| Windows | ❌ | ✅ |
| 工具链耦合 | 严格 | 只约束 C ABI |

### 6.2 ABI 约定（唯一要守的东西）

```c
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT int plugin_interface_version(void);              /* 版本协商 */
EXPORT const char* plugin_name(void);
EXPORT int plugin_run(const char* in, char* out, int out_len); /* 调用方给 buffer */
EXPORT int add(int a, int b);
EXPORT long long bench_sum(int n);
```

规则：

1. **只用 C 类型**：`int` / `long long` / `char*` / 定长 buffer / 函数指针；不要传 Go interface、JS 对象、Python 对象；
2. **字符串用「调用方给 buffer + 长度」**，避免跨语言 new/free 不对称；
3. **导出符号名稳定**，加 `interface_version` 便于演进；
4. **不跨边界抛异常/回调宿主**（复杂场景走 RPC）；
5. **结构体布局**要显式（`#pragma pack`、固定宽度类型），否则不同编译器/语言对不齐；
6. **生命周期**由调用方管理，库不持有宿主对象。

### 6.3 平台映射与加载方式

| 宿主 | Linux | Windows | macOS |
|---|---|---|---|
| 库文件 | `libgreet.so` | `greet.dll` | `libgreet.dylib` |
| Go 加载 | cgo `dlopen/dlsym` | `syscall.NewLazyDLL` + `NewProc`（纯 Go）或 cgo `LoadLibrary` | 同 Linux |
| Node 加载 | N-API addon `.node`（内部 `dlopen`） | N-API addon `.node`（内部 `LoadLibraryA`） | 同 Linux |
| Python 加载 | `ctypes.CDLL` | `ctypes.CDLL` / `WinDLL` | `ctypes.CDLL` |

> Node **不能直接 `require` 裸库**：`require` 只认 JS 模块和 `.node`（N-API addon，`process.dlopen`）。裸 DLL/so 必须包一层 addon 或用 FFI 库（koffi / ffi-napi）。

### 6.4 三语言消费代码

**Go（Linux，cgo dlopen）**

```go
/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
static int (*g_add)(int, int);
static int load(const char* p) {
    g_add = (int(*)(int,int))dlsym(dlopen(p, RTLD_NOW), "add");
    return g_add ? 0 : -1;
}
static int call_add(int a, int b) { return g_add(a, b); }
*/
import "C"

C.load(C.CString("./build/libgreet.so"))
fmt.Println(C.call_add(1, 2))
```

**Go（Windows，纯 Go 无需 cgo）**

```go
var dll = syscall.NewLazyDLL("greet.dll")
var procAdd = dll.NewProc("add")
r, _, _ := procAdd.Call(1, 2)
fmt.Println(int32(r))
```

**Node（N-API addon，跨平台）**

```c
/* addon.c 片段：Init 时加载库、取符号；再暴露成 JS 函数 */
#ifdef _WIN32
  lib = LoadLibraryA(path);   sym = GetProcAddress(lib, name);
#else
  lib = dlopen(path, RTLD_NOW); sym = dlsym(lib, name);
#endif
```

```js
const addon = require("./build/greeter.node");
addon.add(1, 2);        // 3
addon.run("world");     // hello, world (C ABI plugin)
```

**Python（ctypes，无需写 C）**

```python
import ctypes
lib = ctypes.CDLL("./build/libgreet.so")   # Windows: ctypes.CDLL("greet.dll")
lib.plugin_name.restype = ctypes.c_char_p
lib.add.argtypes = [ctypes.c_int, ctypes.c_int]
print(lib.add(1, 2))                       # 3
```

### 6.5 Windows 专项

**Go plugin 在 Windows 不可用**（实测报错）：

```text
$ GOOS=windows GOARCH=amd64 go build -buildmode=plugin -o x.dll .
-buildmode=plugin not supported on windows/amd64
```

但 **Go 可以编译 Windows DLL**：

```bash
GOOS=windows GOARCH=amd64 CGO_ENABLED=1 CC=x86_64-w64-mingw32-gcc \
  go build -buildmode=c-shared -o greet-go.dll .
```

产物特征（实测）：

| | C 实现 `greet.dll` | Go c-shared `greet-go.dll` |
|---|---|---|
| 大小 | 89 KB | 2.7 MB（带 Go runtime） |
| 导出 | `add` `bench_sum` `plugin_interface_version` `plugin_name` `plugin_run` | `Add` `PluginInterfaceVersion` `PluginName` `_cgo_dummy_export` |
| 头文件 | 手写声明 | 自动生成 `greet-go.h` |
| 依赖 | KERNEL32.dll / msvcrt.dll | 更多（Go runtime） |

验证 DLL 可用性的方式（无法在 Linux 执行 Windows DLL 时）：

1. `file build/greet.dll` → PE32+ x86-64；
2. `objdump -p` 看导出表 / 导入表；
3. 编一个 Windows 消费者链接它，检查消费者导入表出现 `DLL Name: greet.dll`；
4. 交叉编译 Go 的 Windows 消费者（`syscall.NewLazyDLL`）；
5. 真机/CI `windows-latest` 跑加载与基准。

Windows 上 Node addon 用 `node-gyp rebuild`（`nodehost/binding.gyp` 已备好，`addon.c` 的 `#ifdef _WIN32` 自动切 `LoadLibraryA`）。

### 6.6 何时用 C ABI

适合：

- 一份算法/协议核心要被 Go、Node、Python 同时复用；
- 希望同进程、低延迟（避免 RPC 的 µs~ms）；
- Windows 部署（Go plugin 不可用）；
- 模块作者愿意遵循 C ABI 约束。

不适合：

- 需要崩溃隔离、热替换、多版本并存 → RPC；
- 需要传递复杂语言对象 / 异常 → RPC 或同语言；
- 模块接口是高频细粒度小函数 → 需要批量接口，否则 30~45 ns/次会累积。

---

## 7. 方案 C：RPC 子进程

**形态**：模块编译成独立可执行文件（Go 编译的 exe / Node 脚本 / Python 模块），宿主通过 stdio / Unix socket / TCP 用 JSON-RPC、gRPC、或自定义协议通信。

**代表**：`hashicorp/go-plugin`（gRPC 双向）、自写 JSON-RPC over stdio。

能力：

- 跨语言、跨平台（Windows 完全可用）；
- 崩溃隔离：模块崩了宿主不受影响，可自动重启；
- 热替换：换进程即可；
- 多版本并存：B 用 D@1、C 用 D@2 各起各的进程；
- 可远程部署（跨机器）。

代价：

- 每次调用数十 µs~数百 µs（序列化 + IPC + 框架开销），比同进程慢 3~5 个数量级；
- 需要进程管理、超时/重试、协议版本、日志/调试；
- 内存：每个模块一个运行时进程（几十 MB 起）。

适合：模块级/服务级调用、插件隔离、Windows 部署、需要独立发布版本。

---

## 8. 方案 D：WASM 与编译期静态方案

**WASM**

- 一份 `.wasm` 跨平台；
- 沙箱隔离好，适合不可信模块/多租户；
- 但系统能力（文件/网络）必须宿主显式提供，生态/工具链成本高，性能通常低于原生（调用亚 µs 级，实例化 ms 级）；
- 适合插件市场、规则引擎、可移植计算内核。

**编译期静态**

- 模块就是库代码，编译进宿主；
- 零运行时开销、类型完整；
- 但无法热更新，改模块要重编宿主；
- 适合稳定的内部模块、不需要独立生命周期。

---

## 9. 能力矩阵

| 能力 | 同语言原生 | C ABI 共享库 | RPC 子进程 | WASM | 编译期静态 |
|---|---|---|---|---|---|
| 跨语言 | ❌（各管各的） | ✅ | ✅ | ✅ | ❌ |
| Windows | Go ❌ / Node/Py ✅ | ✅ | ✅ | ✅ | ✅ |
| 进程内调用 | ✅ | ✅ | ❌ | ✅（VM 内） | ✅ |
| 热更新 | Go ❌ / Node/Py ✅ | 有限 | ✅ | ✅ | ❌ |
| 崩溃隔离 | ❌ | ❌ | ✅ | ✅ | ❌ |
| 复杂类型 | ✅ 原生对象 | ❌ 仅 C 类型 | ✅ 协议（JSON/Proto） | 受限 | ✅ |
| 多版本并存 | 难 | 难（需命名空间/重启） | ✅ 天然 | ✅ 实例隔离 | ❌ |
| 内存效率 | 高（同进程） | 高（代码页共享） | 低（每进程运行时） | 中 | 最高 |
| 分发/构建复杂度 | 低 | 中（每平台产物） | 中高 | 中 | 低 |
| 调试难度 | 低 | 中 | 高 | 中高 | 低 |
| 安全边界 | 无 | 无 | 强 | 强 | 无 |

---

## 10. 性能衰减：实测数据与解读

### 10.1 实测方法

- 环境：WSL2 Linux x86-64，Go 1.22.2，Node 22，Python 3.12，gcc 13；
- 测试函数：`add(int,int)` 与 `plugin_run(char*, char*, int)`（字符串往返）；
- 循环 30 万~5 亿次，取总时间/次数；
- 相对比较用途，**绝对数字换机器会变**。

### 10.2 实测结果（单次调用）

| 路径 | 单次开销 | 相对同语言原生 |
|---|---|---|
| C 直接链接共享库函数 | **1.85 ns** | ≈1×（共享库本体的速度） |
| C `dlopen` + 函数指针 | **1.81 ns** | ≈1× |
| Go 普通函数 | **1.40 ns** | 1× |
| Go `plugin` 函数符号调用 | **1.40 ns** | ≈1× |
| Go `plugin` 接口方法调用 | **2.64 ns** | ≈1.9× |
| Node 普通函数 | **1.2~1.4 ns** | 1× |
| Node ESM 模块方法 | **1.0~1.1 ns** | ≈1× |
| Python 普通函数 / importlib 模块函数 | 百 ns 级 | ≈1×（解释器开销） |
| **Go cgo → C ABI（int）** | **42.6~44.0 ns** | **~30×** |
| **Node N-API → C ABI（int）** | **29.5~33.2 ns** | **~27×** |
| **Go cgo → C ABI（字符串往返）** | **184.5 ns** | **~130×** |
| **Node N-API → C ABI（字符串往返）** | **108.9 ns** | **~90×** |
| **Python ctypes（int）** | **532.6 ns** | **~380×** |
| **Python ctypes（字符串）** | **654.4 ns** | **~470×** |
| **C ABI 批量接口（1 次调用算 1e6 次）** | **0.22~0.25 ns/元素** | ≈1× |
| RPC 子进程（本地回环，典型） | 数十 µs~数百 µs | 10³~10⁵× |
| WASM 调用（典型） | 亚 µs 级 | 10~100× |

### 10.3 加载 / 内存 / 启动

| 指标 | 实测/典型 |
|---|---|
| Go `dlopen` 共享库 | 1.3~2.0 ms（一次性） |
| Node addon + `dlopen` | 7.8~8.1 ms（一次性） |
| `LoadLibrary`（Windows，典型） | 1~3 ms |
| RPC 子进程启动 | 20~200 ms |
| WASM 实例化 | ms 级 |
| C 实现 DLL 体积 | 89 KB |
| Go c-shared DLL 体积 | 2.7 MB（带 runtime） |
| 共享库代码页 | 只读，多进程共享，内存效率高 |

### 10.4 吞吐换算

| 路径 | 约每秒调用次数 |
|---|---|
| C 直接调用 | ~5 亿/s |
| Go cgo int | ~2300 万/s |
| Node N-API int | ~3000 万/s |
| Go cgo 字符串 | ~540 万/s |
| Node N-API 字符串 | ~900 万/s |
| Python ctypes int | ~190 万/s |
| RPC（100 µs/次） | ~1 万/s |

1e6 次调用耗时：Go cgo int ≈ 43 ms；Node N-API int ≈ 33 ms；批量接口 1e6 元素 ≈ 0.24 ms（约 150 倍差距）。

### 10.5 怎么解读

1. **同语言原生几乎零衰减**（ns 级），是性能上限；
2. **C ABI 的 30~45 ns 来自 FFI 桥（cgo / N-API 栈切换与参数转换），不是 DLL 本身**；DLL 函数本体 ≈ 1~2 ns；
3. **字符串/对象编组是更大的成本**（100~650 ns/次），尤其 Python ctypes、Go `C.CString`；
4. **批量接口能摊回原生速度**：一次调用处理 1e6 个元素 ≈ 0.24 ns/元素；
5. **RPC 直接跳 3 个数量级**，适合模块级调用，不适合高频小函数；
6. 真实业务中 IO/序列化/编解码通常远大于边界开销，**只有「每秒百万级小函数调用」才会被 FFI 卡住**。

### 10.6 优化清单

- 粗粒度 API：一次传一批数据（数组/缓冲区），C 里循环；
- 避免每次跨边界新建字符串/JSON：复用 buffer，或用二进制数组（Node typed array / Go `unsafe.Slice`）；
- 句柄缓存：`dlopen`/`LoadLibrary` 只做一次；
- Go：避免热路径 `C.CString`；减少 cgo 调用次数；必要时用批量接口；
- Node：N-API 创建字符串/对象是主要成本，数据用 `napi_create_external_buffer` / typed array；
- Python：ctypes 开销大，追求性能用 cffi 编译模式或批量接口；
- 高频场景考虑把逻辑整体下沉到 C 侧，宿主只做一次调用。

### 10.7 Windows 预期

- Go 用 `syscall.NewLazyDLL` + `Proc.Call`（纯 Go `syscall.SyscallN`，无需 cgo），开销与 cgo 同量级（几十 ns），可能略轻；
- Node 用 N-API addon（MSVC 编译，`LoadLibraryA` 加载），单次调用与 Linux 的 N-API 同量级（~30 ns）；
- DLL 与 so 的差别只在加载器与调用约定，函数体执行时间一致；
- 精确数字需在 Windows 实机/`windows-latest` CI 实测（当前 Linux 环境无 wine，无法执行 Windows DLL）。

---

## 11. 依赖关系：菱形依赖与模块间依赖

### 11.1 菱形依赖本身不是问题

```
        A
       / \
      B   C
       \ /
        D
```

在「分支只改自己目录、merge 进同一仓库」的模式下：

- A/B/C/D 各有目录，merge 互不冲突，**merge 顺序无关**；
- D 在 main 上只有一份，**源码层面天然去重**；
- 所以「菱形依赖导致两份 D」在这个模式里不存在。

### 11.2 真正会出问题的四个点

| 风险 | 场景 | 后果 |
|---|---|---|
| 版本冲突 | B 要 D@1，C 要 D@2 | 单版本无法同时满足 |
| 实例重复 | loader 按目录/URL 分别加载 D | B、C 各拿一份 D，状态不共享 |
| 初始化顺序 | D 未初始化，B/C 先用 | 拿到未初始化依赖 |
| 更新不级联 | D 更新，B/C/A 仍持旧引用 | 一直用旧代码 |
| 环依赖 | A→B→A | 加载死循环，必须拒绝 |

### 11.3 各方案下的依赖表现

- **同语言原生**：source/loader 单实例易保证；版本冲突由包管理器/构建解析；
- **C ABI**：同进程通常只有一份库；多版本需命名空间/符号隔离或重启；无依赖图概念，调用方自理；
- **RPC**：各进程独立，**最适合 B 用 D@1、C 用 D@2**；
- **WASM**：实例隔离，多版本容易；
- **静态**：构建期解析依赖，运行期无问题，但无热更新。

### 11.4 推荐策略

1. **首选：D 做成「库」而不是「模块」**——共享代码放仓库根（Go package / npm workspace / Python package），由宿主统一构建；模块只留功能叶子，**禁止模块间直接 import**；
2. 确实要模块间依赖：`module.json` 加 `dependencies` + 版本约束；**全局唯一 lock**；loader 做**拓扑排序 + 单实例缓存 + 环检测**；sync 做 **affected 闭包重建**（D 更新 → {B,C,A} 重建/重载）；
3. 需要同一依赖多版本：把 D 做成 **RPC 服务/独立进程**（或 WASM 实例），按接口版本兼容；
4. 跨语言依赖：C ABI 库或 RPC；
5. 版本不兼容时**直接报错**，不要静默取其中一个。

affected 闭包示例：

```
D 变了 → 反向依赖图 {D: [B, C], B: [A], C: [A]} → {B, C, A}
       → 按拓扑序重新构建/加载：D → B/C → A
```

---

## 12. 版本、锁定、回滚、可观测性

- **lock 文件** `modules.lock.json`：记录每个模块的 `name/language/branch/commit/version/artifact`，是审计与对账的锚点；
- 可扩展：产物 `sha256`、同步触发者、同步时间、依赖图快照；
- **版本钉住**：`modules.json` 支持 `ref`（tag/分支）替代 `branch`，或加 `pin_commit`；
- **回滚**：`git revert <merge commit>`；注意下次 sync 会合回最新；要长期回滚用 ref/pin；
- **可观测**：宿主启动打印 `name/version/commit`；sync 每次变更打印一行（谁、哪个分支、新旧 commit）；CI 保留日志；
- **测试门禁**：同步后跑模块自带测试 + 宿主冒烟（加载全部模块、调用一次 `run`）；失败则不同步该模块（`sync.py --test` 钩子）；
- **失败处理**：merge 冲突 abort、构建失败保留旧 artifact、lock 不更新、告警。

---

## 13. 安全与隔离

- 能动态加载的模块 = **能执行任意代码**；来源必须受信（PR review + CI + 受保护分支）；
- 同步机器人只信任 `origin` 上的受保护分支；产物可签名/校验 sha256；
- **进程内方案（plugin / C ABI）没有隔离**：模块崩溃/野指针会拖垮宿主；
- **RPC / WASM 有隔离**：适合不可信模块、多租户；
- 模块依赖的第三方库要锁定版本；
- 生产环境不要让进程自己去拉 git。

---

## 14. 分发与构建矩阵

### 14.1 按平台出产物

```text
plugin.c ──┬── Windows (MinGW/MSVC) → greet.dll
           ├── Linux (gcc)          → libgreet.so
           └── macOS (clang)        → libgreet.dylib

addon.c  ──┬── node-gyp/MSVC (win32-x64) → greeter.node
           ├── node-gyp (linux-x64)       → greeter.node
           └── node-gyp (darwin-arm64)    → greeter.node
```

> `.dll` 是 PE 格式，**Linux 不能复用**，必须编 `.so`；反过来也一样。源码/接口/宿主调用代码可以完全一样。

### 14.2 产物布局

```text
dist/
├── win32-x64/    greet.dll     greeter.node
├── linux-x64/    libgreet.so   greeter.node
└── darwin-arm64/ libgreet.dylib greeter.node
```

### 14.3 CI 矩阵（示例）

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
steps:
  - run: gcc -shared -fPIC -O2 -o libgreet.so plugin.c    # Linux/macOS
  - run: cl /LD plugin.c /Fe:greet.dll                     # Windows（或用 MinGW）
  - run: npx node-gyp rebuild                              # .node
  - run: bash abi-plugin/run.sh                            # 各平台自测
```

### 14.4 Node 预编译分发

npm 生态常用 `optionalDependencies` + `os`/`cpu` 字段发布各平台 `.node`（napi-rs 模式），运行时按平台选择目录，避免用户本地编译。

### 14.5 Python 分发

- `ctypes` 不需要编译包装，但要随包分发对应平台的 `.so/.dll`；
- 可用 `cibuildwheel` / 平台 wheel 分发；
- 或用 `cffi` 生成编译扩展（更快但需要编译）。

---

## 15. 选型决策树

```
需要模块化扩展？
├─ 否 → 单体/编译期静态
└─ 是
   ├─ 同语言、同平台、性能敏感？
   │    └─ 是 → 同语言原生（Go plugin / ESM / importlib）
   ├─ 需要一份二进制跨语言共享？
   │    └─ 是 → C ABI 共享库（.so/.dll/.dylib）
   ├─ Windows 上跑 Go 模块？
   │    └─ 是 → C ABI DLL（NewLazyDLL）或 RPC（plugin 不可用）
   ├─ 需要热更新 / 崩溃隔离 / 多版本并存？
   │    └─ 是 → RPC 子进程
   ├─ 不可信模块 / 多租户 / 强沙箱？
   │    └─ 是 → WASM 或独立进程
   └─ 只需最快跑通、后续可换？
        └─ 同语言原生 + 保持 module.json 契约不变
```

### 推荐组合

| 场景 | git 层 | 加载层 |
|---|---|---|
| 单语言内部扩展 | merge + 同步器 | 同语言原生 |
| 跨语言算法/协议核心 | merge + 同步器 | C ABI 共享库 |
| Windows 部署的 Go 模块 | merge + 同步器 | C ABI DLL / RPC |
| 插件市场/多租户 | merge + 同步器 | WASM / RPC |
| 稳定内部模块 | merge + 同步器 | 编译期静态 |
| 存量乱分支迁移 | subtree 或重建模块分支 | 按上面选 |

---

## 16. 本实验仓库结构与命令

```
branch-module-lab/
├── README.md                 # 快速开始
├── HANDBOOK.md               # 本文
├── DESIGN.md                 # 设计文档（方案对比、迁移步骤）
├── modules.json              # 模块来源登记（分支 → 模块）
├── modules.lock.json         # 生成物：commit / version / artifact
├── sync.py                   # 同步器：status / once / watch
├── go/
│   ├── sdk/sdk.go            # Go 模块契约
│   └── host/main.go          # Go 宿主（plugin + re-exec）
├── js/host.mjs               # Node 宿主（ESM import + 破缓存热重载）
├── python/host.py            # Python 宿主（importlib + mtime 热重载）
├── modules/<lang>/<name>/    # 由 feature 分支 merge 出来的模块
├── demo/
│   ├── author_modules.py     # 模拟模块作者（建分支/升级/push）
│   └── demo.sh               # 端到端演示（v1 → v2 → v3 全自动）
├── ci/sync-modules.yml       # CI 示例
└── abi-plugin/
    ├── plugin.c              # C ABI 实现（.so/.dll 同一份源码）
    ├── gohost/               # Go 消费者（cgo dlopen）
    ├── gohost-win/           # Go 消费者（syscall.NewLazyDLL）
    ├── nodehost/             # N-API addon + binding.gyp + bench
    ├── go-export/            # 反向：Go -buildmode=c-shared 产 DLL
    ├── run.sh                # 构建 + 运行 + 交叉编译验证 + 基准
    └── README.md             # C ABI 专题文档
```

命令：

```bash
cd branch-module-lab

# 1) 分支 → 模块 端到端演示（会重置本实验仓库到 demo-base tag）
bash demo/demo.sh

# 2) 手动同步
python3 sync.py status          # 查看同步状态
python3 sync.py once            # 合并 + 构建 + lock
python3 sync.py watch --interval 2

# 3) 三语言宿主
go build -o .bin/go-host ./go/host && ./.bin/go-host --root .
node js/host.mjs --root .
python3 python/host.py --root .

# 4) C ABI 共享库（Go / Node；Python 用 ctypes）
bash abi-plugin/run.sh
```

---

## 17. 常见误解与坑清单

**误解**

1. ❌「Go plugin 就是 DLL」→ 它只能加载 Go 编译的 `.so`，不是通用 DLL 加载器；
2. ❌「Node 能直接 require DLL」→ 必须 N-API addon 或 FFI 库；
3. ❌「Windows 上也能用 Go plugin」→ 不支持，实测报错；
4. ❌「C ABI 能传 Go interface / JS 对象」→ 只能 C 类型；
5. ❌「merge 一次模块就自动更新」→ 需要持续同步；
6. ❌「submodule 能实现自动更新」→ 它是钉版本，方向相反；
7. ❌「菱形依赖一定有两份 D」→ 同仓 merge 模式源码只有一份，问题在版本/实例/加载顺序；
8. ❌「通用方案没有代价」→ C ABI 只有 C 类型且无隔离；RPC 慢 3~5 个数量级；
9. ❌「DLL 到了 Linux 能直接用」→ 必须编 `.so`；
10. ❌「Python 用 DLL 要写包装」→ `ctypes` 直接加载。

**坑清单**

- Go plugin：同工具链、Linux/macOS、不可卸载、原子替换产物、共享接口包；
- Node ESM：缓存破 URL、旧实例不释放、依赖解析范围；
- Python importlib：模块名唯一、`sys.modules`、包内相对导入、重载语义；
- C ABI：字符串 buffer、结构体布局、异常/回调、跨语言 free、平台产物；
- 同步器：幂等、分支不存在跳过、冲突 abort、lock 与 merge 分开提交、main 保护；
- CI：git 身份配置、fetch depth 足够（`fetch-depth: 0`）、分支保护；
- Windows：MinGW/MSVC 编译、`.node` 按平台、`LoadLibrary` 路径、x64 调用约定。

---

## 18. 附录

### 18.1 术语

| 术语 | 含义 |
|---|---|
| ABI | 二进制接口约定（调用约定、类型布局、符号命名） |
| C ABI | 以 C 类型/调用约定为边界的稳定二进制接口 |
| FFI | Foreign Function Interface，跨语言函数调用桥（cgo、N-API、ctypes 等） |
| plugin（Go） | Go 标准库动态加载包，加载 Go 编译的 `.so` |
| N-API | Node 原生扩展的稳定 ABI，产物 `.node` |
| dlopen/LoadLibrary | Linux/Windows 运行时加载共享库的 API |
| RPC | 远程过程调用（这里指进程间调用） |
| 幂等同步 | 重复执行结果一致的同步循环 |
| lock 文件 | 记录模块当前 commit/version/产物的生成物 |
| 菱形依赖 | A→B→D、A→C→D 的依赖图 |

### 18.2 实测原始数据（摘要）

```text
# 同语言原生
pure Go func            1.40 ns/op
Go plugin 函数符号       1.40 ns/op
Go plugin 接口方法       2.64 ns/op
pure JS func            1.2~1.4 ns/op
ESM 模块方法             1.0~1.1 ns/op

# C ABI（本机 Linux .so；DLL 走同一 FFI 桥）
C 直接链接               1.85 ns/op
C dlopen 函数指针        1.81 ns/op
Go cgo int              42.6~44.0 ns/op
Node N-API int          29.5~33.2 ns/op
Go cgo 字符串            184.5 ns/op
Node N-API 字符串        108.9 ns/op
Python ctypes int       532.6 ns/op
Python ctypes 字符串     654.4 ns/op
批量（1e6/次调用）       0.22~0.25 ns/元素

# 加载
Go dlopen                1.3~2.0 ms
Node addon + dlopen      7.8~8.1 ms

# 体积
greet.dll (C)            89 KB
greet-go.dll (Go c-shared) 2.7 MB
```

### 18.3 相关文件

- [DESIGN.md](DESIGN.md)：设计文档（git 层方案对比、迁移步骤、三语言细节）
- [abi-plugin/README.md](abi-plugin/README.md)：C ABI 专题（实测、Windows 说明）
- [README.md](README.md)：实验快速开始
- [demo/demo.sh](demo/demo.sh)：分支→模块端到端演示
- [sync.py](sync.py)：同步器实现
- [go/host/main.go](go/host/main.go) · [js/host.mjs](js/host.mjs) · [python/host.py](python/host.py)：三语言宿主
- [abi-plugin/plugin.c](abi-plugin/plugin.c) · [abi-plugin/nodehost/addon.c](abi-plugin/nodehost/addon.c)：C ABI 与 N-API 包装