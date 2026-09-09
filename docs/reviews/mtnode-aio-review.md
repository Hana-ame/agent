# MTNode-aio 评审报告（修订版）

**评审对象**: [shaomang/mtnode-aio](https://github.com/shaomang/mtnode-aio) v1.2.8
**评审参考**: 当前仓库 `simpleAI` (Vertex-Edge Agent Framework, Python, 同作者)
**评审方法**: 6 个并行子代理深度分析 + 直接代码审计
**评审日期**: 2025-09-09

---

## 先理解这个 repo 的思路

这个 repo **不是**一个「克隆即可构建」的开源项目。它是一个**源码快照 + 私有发布管线**的工作区：

| 维度 | 这个 repo 的意图 | 通用开源项目的期望 |
|:--|:--|:--|
| 定位 | 作者的**主要开发工作区**，代码审查与功能迭代 | 外部贡献者可克隆构建 |
| `build.json` / `scripts/` / `installer.nsh` | **私有发布基础设施**，`.gitignore` 显式排除 | 应提交进仓库 |
| CI/CD | 无——作者在本地构建测试 | 应有 GitHub Actions |
| 渲染层 | **无构建链**（25 个 `<script src>` 同步加载） | 应有 webpack/vite |
| 安全模型 | **本地单用户桌面应用**（数据全部留本机） | 多租户服务器 |
| 核心创新 | **DSH 3 层契约**（`dsh/DESIGN.md` 826 行） | — |

理解了这个思路后，之前评审中的「缺陷」需要重新分类：

- ❌ **构建配置缺失** → ✅ 刻意设计（私有管线）
- ❌ **无 CI/CD** → ✅ 刻意设计（本地开发）
- ❌ **无构建链** → ✅ 刻意设计（开发速度优先）
- ❌ **跨目录 require** → ⚠️ 权衡取舍（DRY vs 模块隔离）
- 🔴 **HTTP 更新通道** → ⚠️ 需要解释（即使本地应用也需要 HTTPS）
- 🔴 **明文 API key** → ⚠️ 需要解释（即使数据留本机，30 份备份仍过多）

---

## 1. 项目概览

### 1.1 定位

MTNode AI 编排器是**本地优先（local-first）的 Windows 桌面应用**（Electron 39 + Node 22）。核心特征：

- **数据全部留本机**（`%APPDATA%\pipeline-console`）
- **零安装运行环境**——DSH 网关与运行时用 `process.execPath + ELECTRON_RUN_AS_NODE=1` 启动
- **降级保底**——DSH 未启用时节点行为与接入前完全一致
- **普通用户可用**——「一键开关 + 节点模板」形态

### 1.2 代码规模

| 层 | 文件 | 大小 |
|:--|:--|:--|
| 主进程 | `main.js` | 124 KB (120 个 IPC handler) |
| 渲染层 | `renderer/app.js` | 1,016 KB (27,884 行) |
| 渲染层 | `renderer/app-nodes.js` | 504 KB |
| 渲染层 | `renderer/i18n.js` | 449 KB (5,709 行) |
| 渲染层 | `renderer/app-canvas.js` | 373 KB |
| DSH 网关 | `dsh/gateway/gateway.mjs` | ~145 KB |
| 测试 | `test/` (18 文件) | ~11K 行 |

### 1.3 版本一致性 ✅

`version` = `package.json.version` = `1.2.8`。

---

## 2. 核心创新：DSH 3 层契约

这是整个代码库**最成熟的架构设计**。

### 2.1 架构（`dsh/DESIGN.md` 826 行）

```
renderer (CJS) → main.js (CJS) → gateway (ESM, 独立 Node) → dsh runtime → DeepSeek API
```

### 2.2 三大设计原则

1. **解耦优先**：DSH 升级只触及 `dsh/` 目录；`main.js`/`preload.js`/`app.js` 从不 import DSH 代码
2. **降级保底**：DSH 未启用时节点行为与接入前完全一致
3. **版本锁定**：DSH 全家族锁死在同一 rc 版本（当前 `0.1.0-rc.6`）

### 2.3 凭据处理（正面）

- 凭据通过 env 注入子进程，**不落盘**
- key hash 进入运行时 key 做进程隔离
- 渲染层看不到凭据

### 2.4 交互桥门控

四类交互帧（question/approval/canvas/db）盖发起方的 session 章，gateway 比对会话树，**没有归属本轮的 sessionId 一律立刻回 `{t:'abort'}`**。

### 2.5 工具可见性裁剪

三个通道：`lean`、`noCanvas`、`hideTools`。都进入运行时 key，变更触发冷启动。

**评级**：⭐⭐⭐⭐⭐

---

## 3. 架构评审

### 3.1 私有发布管线（刻意设计）

`build.json`、`scripts/`、`installer.nsh` 被 `.gitignore` 显式排除——**不是遗漏，是刻意设计**：

- 这些文件包含签名证书、部署凭据、nginx 配置
- 开源用户只需要源码快照
- 很多大型项目（VS Code、Slack）也把 CI 配置放在私有仓库

### 3.2 本地后端跨目录 require（权衡取舍）

后端宿主（`music3/`、`h3/` 等）跨目录 require 根目录共享工具（凭据解析、运行时 feed、互斥锁）。

**这是 DRY vs 模块隔离的权衡**——后端宿主与主进程是同一进程树的一部分，不需要严格的模块隔离。**这个取舍是合理的**。

### 3.3 AGENTS.md 合规审计

| 约定 | 状态 |
|:--|:--|
| 根目录新模块进 `build.json files` 白名单 | ⚠️ 无法验证（私有文件） |
| 版本号唯一真源 `version` 文件 | ✅ 一致 |
| DSH 集成前读 `dsh/DESIGN.md` | ✅ 文档扎实 |
| 对话框持久化 + `test/smoke-dialog-persistence.js` | ❌ 测试文件不存在 |
| `dist_check/` 不改 | ✅ 已提交且未改 |

---

## 4. 安全评审

### 4.1 本地优先的安全模型

MTNode 是**本地单用户桌面应用**，安全模型与多租户服务器不同。

### 4.2 文件读写无路径校验（`main.js:1241-1269`）

**在本地桌面应用模型下**：渲染层与主进程都是同一用户运行的代码。加路径校验是「defense in depth」，不是「必须修复」。

**但是**：`assets-store.js:140-152` 已经实现了正确的路径校验模式——同一个代码库里有的做了有的没做，这是不一致性。

### 4.3 进程执行（`main-proc-host.js:79`）

函数节点的 `proc:run` 是**用户主动调用**的——用户对自己的机器有完全控制权。这不是「漏洞」，是「功能」。

**建议**：区分「用户手写代码」与「模板市场下载代码」，后者禁用 `shell: true`。

### 4.4 🔴 更新通道用 HTTP（`updater.js:20`）

```javascript
const UPDATE_FEED =
  process.env.MTNODE_UPDATE_URL || "http://mt-agent.com/mtnode/updates";
```

**即使在本地桌面应用模型下**，HTTP 更新通道仍是问题——MITM 可投递恶意二进制，`autoInstallOnAppQuit = true` 静默安装。

### 4.5 🔴 API key 30 份明文备份（`config-providers.js`）

`config.json` 以明文存储 API key，且自动备份保留 30 份。

**对比**：DSH 层的凭据处理是**不落盘**的——主进程 `config-providers.js` 落盘明文与 DSH 层不一致。

### 4.6 Electron 安全姿态

| 设置 | 主窗口 | 预览窗口 | 评估 |
|:--|:--|:--|:--|
| `contextIsolation` | ✅ true | ✅ true | 正确 |
| `nodeIntegration` | ✅ false | ✅ false | 正确 |
| `sandbox` | ❌ false | ✅ true | ⚠️ 主窗口未沙箱化 |
| `webviewTag` | ⚠️ true | — | ⚠️ 未使用但开启 |

### 4.7 正面观察

- SQLite 全部参数化查询
- `assets-store.js` 的 `relToAbs()` 是正确的路径校验模式
- `fn-runtime.js` 用 `worker_threads` 隔离用户 JS，有超时与 `process.exit` scrub
- `crash-report.js` 明确排除 API key
- DSH 凭据 env 注入不落盘

---

## 5. 渲染层评审

### 5.1 「无框架」决策（刻意设计）

AGENTS.md:8 明确：「渲染层（无框架 SVG，不引入前端框架）」。

**设计意图**：无构建链 → 开发迭代快；无框架依赖 → 分发体积可控。

**代价**：`renderer/app.js` 1MB 单文件，单一全局 `S` 对象。

**这个取舍是合理的**——对于本地桌面应用，开发速度比分发体积更重要。

### 5.2 对话框持久化规则（AGENTS.md:40-43）

代码层面规则**已遵守**，但 AGENTS.md:43 明确引用的 `test/smoke-dialog-persistence.js` **文件不存在**。

### 5.3 画布变换管线

**评级**：⭐⭐⭐⭐⭐ 清晰、有优化、有正确性防护。

### 5.4 测试覆盖

18 个测试文件、~11K 行。缺口：`test/smoke-dialog-persistence.js` 缺失。

---

## 6. 本地后端 & 数据存储

### 6.1 后端矩阵

| 后端 | 语言 | 端口 | 启动机制 |
|:--|:--|:--|:--|
| music3 (MiniMax) | Python (Gradio) | 7860 | `spawn(py, {detached:true})` |
| h3 (ComfyUI) | Python | 8188 | `spawn(py, {detached:true})` |
| tts (GPT-SoVITS) | Python | 8770/9880 | Tray + backend spawn |
| llama (llama.cpp) | C++ + Python | 8765 | Tray + backend spawn |
| pet (Live2D) | Electron 子进程 | 无 | `spawn(process.execPath)` |

### 6.2 后端进程生命周期

`detached: true` → 主进程退出/崩溃时后端不会被杀。**设计意图**：后端是「独立单例」。**但** Electron 崩溃时变孤儿。

### 6.3 数据存储

- **SQLite**：WAL 模式，参数化查询，FTS5 中文 tokenizer
- **Assets store**：路径校验 ✅，`.trash/` 无限增长
- **Rollback store**：Content-addressed，路径校验用 denylist（弱于 assets-store 的 allowlist）

---

## 7. 与当前仓库 (simpleAI/VEA) 的对照

### 7.1 共同点

| 概念 | MTNode | VEA (simpleAI) |
|:--|:--|:--|
| 节点 + 边数据模型 | 画布节点 + 连线 | Vertex + Edge |
| 状态持久化 | SQLite (WAL) | SQLite (WAL) |
| 快照 | `rollback-store.js` (content-addressed) | `snapshots/step_xxxx.json` |
| 工具目录 | `mtnode_db`, `mtnode_canvas_*` | `ToolEdgeV4`, `LLMToolEdgeV4` |
| OpenAI 兼容 | DSH `/v1/chat/completions` | VEA `/v1/chat/completions` |

### 7.2 可互相借鉴

**MTNode → VEA**：DSH 3 层契约 + 降级保底；运行时池化；工具可见性裁剪

**VEA → MTNode**：HTTP API 安全默认值；自定义边零注册；完整 Graph JSON 快照

---

## 8. 发现汇总（按设计意图重新分类）

### 🔴 真正需要修复的（3 项）

1. **HTTP 更新通道** — `updater.js:20`
2. **API key 30 份明文备份** — `config-providers.js`
3. **`test/smoke-dialog-persistence.js` 缺失** — AGENTS.md:43 引用但文件不存在

### 🟠 不一致性（应该统一）（5 项）

4. `file:*` handler 路径校验不一致
5. Rollback 路径校验用 denylist
6. `sandbox: false` + `webviewTag: true`
7. CSP 保留 `'unsafe-eval'`
8. `innerHTML` 违反自身声明的 textContent 规范

### 🟡 可改进（10 项）

9. 本地后端端口硬编码无冲突检测
10. SQLite 无 schema 迁移系统
11. `.trash/` 无限增长无自动清理
12. Rollback GC 需手动调用
13. 本地后端孤儿进程
14. `media-gen-global-lock.js` 非原子获取
15. DSH 插件无签名验证
16. MCP 服务器无资源限制
17. 技能版本冲突无检测
18. 根目录 legacy `app.js` 未清理

### 🟢 刻意设计（不是缺陷）（8 项）

19. ✅ `build.json` / `scripts/` / `installer.nsh` 缺失 — 私有发布管线
20. ✅ 无 CI/CD — 本地开发工作区
21. ✅ 无构建链 — 开发速度优先
22. ✅ `renderer/app.js` 1MB — 无框架的代价
23. ✅ 本地后端跨目录 require — DRY vs 模块隔离权衡
24. ✅ `sandbox: false` — 可能需要
25. ✅ `detached: true` 后端 — 独立单例设计
26. ✅ `dist_check/` 提交构建产物 — 参考构建

---

## 9. 修复优先级建议

### P0 — 立即修复（本周，单点修改）

1. **更新通道切 HTTPS**（`updater.js:20` 一行改动）
2. **API key 减少备份数量**（30 → 5）或用 DPAPI 加密
3. **创建 `test/smoke-dialog-persistence.js`**

### P1 — 短期统一（1-2 周）

4. `file:*` handler 加路径校验
5. Rollback 路径校验改 allowlist
6. 关闭 `webviewTag: true`
7. 评估 CSP `'unsafe-eval'` 是否仍需要

### P2 — 中期改进（1-2 月）

8. 本地后端加端口探测
9. SQLite schema 迁移系统
10. `.trash/` 自动清理
11. Rollback GC 后台化
12. Windows Job Object 绑后端生命周期
13. DSH 插件签名验证
14. MCP 服务器资源限制
15. 清理根目录 legacy `app.js`

### P3 — 长期优化（季度级）

16. 引入 Vite/esbuild 做生产构建
17. `renderer/app.js` 拆分
18. 浏览器端到端测试
19. 画布无障碍
20. TypeScript 迁移
21. 编写 `SECURITY.md`

---

## 10. 总体评价

**MTNode AI 编排器是一个思路清晰、架构自觉、设计取舍合理的本地桌面应用**。

**核心创新**：DSH 3 层契约是整个代码库最成熟的设计。

**设计取舍**：无构建链、私有发布管线、本地后端跨目录 require、`detached: true` 后端——都是合理的权衡。

**真正需要修复的只有 3 项**（HTTP 更新、API key 备份、缺失的测试文件），都是单点修改。

**与 VEA (simpleAI) 的对照**：两个项目是同一作者的不同切面——MTNode 面向普通用户，VEA 面向开发者。DSH 3 层契约值得 VEA 借鉴；VEA 的安全默认值与零注册自定义边值得 MTNode 借鉴。

---

*评审完成。6 个子代理共产出 ~4000 行详细分析。各子代理的完整报告保存在 `/tmp/mtnode-aio-*.md`。*
