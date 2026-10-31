# git-bridge

一个本地 HTTP 服务，把**一个 git 仓库**公开给**很多 agent** 同步修改。
**Node.js** 实现，**零依赖**（只用 Node 18+ 内置模块），**队列串行处理**所有 git 操作（彻底避免并发竞争）。

每个 agent 只要做一件事：把自己产生的 patch（`git diff` 输出）`POST` 过来，指定一个 branch，服务端在 FIFO 队列里依次 **fetch → apply → commit → push** 到那个 branch。push/fetch 复用本机 `gh` 的凭据。每次 POST 都记入**持久化台账**，可查状态、可驱动下游 CI。

```
agent A ─┐
agent B ─┼── POST /commit?branch=xxx&message=...  (body = unified diff)
agent C ─┘                │
                          ▼
                   ┌──────────────┐
                   │  git-bridge  │  FIFO 队列（单 worker，串行）
                   │  (gh auth)   │  fetch → checkout -B → apply → commit → push
                   └──────────────┘
                          │  每次结果写入 ledger.jsonl
                          ▼
                   GitHub origin      GET /commits  查状态 / 驱动 CI
```

## 设计要点

- **队列串行**：所有 POST 进一个 FIFO 队列，单 worker 依次处理 → 任意时刻只有一个 git 操作在跑，**没有并发竞争**，不需要 worktree / 锁。不同 agent 往不同 branch 提交也安全排队。
- **每次基于最新 remote**：每个 job 都 `fetch` 后 `checkout -B <branch>` 到 `origin/<branch>`（新分支则 `origin/<base>`），再 apply。push 遇到 non-fast-forward 会自动 fetch + 重放（最多 3 次）。
- **台账**：每个 POST 一条记录（id / branch / message / sha / status / 耗时 / 错误），持久化到 JSONL，重启不丢；HTTP 可查，下游 CI 可轮询。
- **gh 鉴权**：启动跑 `gh auth setup-git`，`git push/fetch` 复用 gh 凭据（SSH / HTTPS 都行）。SSH push 不依赖它也能工作。
- **防挂起**：所有 git 子进程 stdin 设为 `/dev/null`，并设 `GIT_TERMINAL_PROMPT=0` / `GIT_SSH_COMMAND=ssh -o BatchMode=yes`，任何需要交互的命令会**立即失败**而不是卡死。

## 前置条件

```bash
gh auth login          # 已登录即可
node --version         # >= 18
git --version
```

## 启动

```bash
# 指向一个已存在的本地 clone（推荐，最省事）
node server.js --repo /path/to/your/repo --base main

# 或让服务端自己 clone
node server.js --repo ./myrepo --repo-url git@github.com:you/repo.git --base main

# 实测用的 agent 仓库
node server.js --repo /tmp/agent-bridge --repo-url git@github.com:Hana-ame/agent.git --base main --port 8765
```

参数（均可改用同名 `GIT_BRIDGE_*` 环境变量）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--repo` | `./repo` | 要操作的本地仓库路径 |
| `--repo-url` | 空 | `--repo` 不存在时先 clone 这个 URL |
| `--base` | `main` | 新 branch 的默认起点 |
| `--host` / `--port` | `127.0.0.1` / `8765` | 监听地址 |
| `--token` | 空 | 设了之后请求需带 `Authorization: Bearer <token>` |
| `--author` | 自动推断 | 默认作者，`Name:email` 格式 |
| `--ledger` | `<repo>.ledger.jsonl` | 台账文件 |
| `--wait-timeout` | `60000` | 同步 POST 等结果的最长时间，超时返回 202 让你轮询 |

默认作者推断顺序：`--author` > `GIT_BRIDGE_AUTHOR_*` > 仓库 `user.name/email` > `gh api user`（`<login>@users.noreply.github.com`）。

## API

### `POST /commit` —— 入队，apply + commit + push

两种 body：

**A. 原始 patch（推荐）**

```bash
git diff > my.patch
curl -sS -X POST \
  "http://127.0.0.1:8765/commit?branch=agent-1/fix-typo&message=fix%20a%20typo" \
  --data-binary @my.patch
```

可选 query：`base=<起点远端分支>`、`async=1`（立即返回 202+id，再轮询）、`notify=<url>`（完成后把结果 POST 到该 URL，驱动下游 CI）。
可选 header：`X-Author-Name` / `X-Author-Email` / `X-Message`。

**B. JSON**

```bash
curl -sS -X POST "http://127.0.0.1:8765/commit" \
  -H 'Content-Type: application/json' \
  -d '{"branch":"agent-1/fix-typo","message":"fix a typo","base":"main",
       "author":{"name":"agent-1","email":"a1@x"},
       "patch":"--- a/f\n+++ b/f\n@@ -1 +1 @@\n-x\n+y\n",
       "notify":"https://ci.example.com/hook"}'
```

默认是**同步**模式：阻塞到 job 完成（最长 `--wait-timeout`），直接返回结果；超时则返回 `202` + `id`，用 `GET /commits/<id>` 轮询。`async=1` 则始终立即返回 `202`。

成功返回：

```json
{"id":"e7bd9b5f","ok":true,"status":"pushed","branch":"agent-1/fix-typo",
 "sha":"3bbb3a6...","short":"3bbb3a6","pushed":true,"attempts":1,"elapsedMs":12872}
```

- patch 干净但无改动 → `"status":"noop","committed":false`
- patch 应用失败 → `400`，`detail` 是 `git apply --check` 的输出
- push 失败 → `502`

### `GET /commits` —— 台账列表（查状态 / 驱动 CI）

```bash
curl -sS "http://127.0.0.1:8765/commits?branch=agent-1/fix-typo&status=pushed&limit=20"
```

返回 `{count, commits:[...]}`，最新在前。`status` 取值：`queued` / `working` / `pushed` / `noop` / `failed`。

### `GET /commits/<id>` —— 单条详情，可选查 GitHub CI

```bash
curl -sS "http://127.0.0.1:8765/commits/e7bd9b5f?checks=1"
```

`?checks=1` 会用 `gh api repos/<owner>/<repo>/commits/<sha>/check-runs` 拉取该 commit 的 GitHub CI 状态（best-effort），附加 `checks: {total, runs:[{name,status,conclusion,url}], summary}`。`<id>` 也支持用 sha / short sha 查。

### `GET /queue`

`{pending, working, current, processed}` —— 队列深度与 worker 状态。

### `GET /health`

仓库 HEAD、origin URL、`gh auth` 是否 OK。

### `GET /branches`

`origin` 下的远端分支列表。

### `GET /`

服务自述 + 用法。

## 给 agent 的典型用法

```bash
# 1. 在自己的工作区改代码
git diff > /tmp/chg.patch

# 2. 一行提交到任意 branch，自动 push（同步，直接拿到 sha）
curl -sS -X POST \
  "http://127.0.0.1:8765/commit?branch=$(hostname)-$(date +%s)&message=wip" \
  --data-binary @/tmp/chg.patch

# 或者异步 + 让 CI 通过 notify webhook 被动接收
curl -sS -X POST \
  "http://127.0.0.1:8765/commit?branch=feat-x&message=wip&async=1&notify=https://ci/hook" \
  --data-binary @/tmp/chg.patch
```

## 下游 CI 怎么接

- **轮询**：CI/orchestrator 定时 `GET /commits?status=pushed&limit=N`，拿到新 `(branch, sha)` 后触发对应构建。
- **webhook**：POST 时带 `notify=<url>`，job 完成后 server 把完整结果 POST 过去（含 sha/branch/status），CI 被动触发。
- **GitHub 侧**：push 本身就会触发仓库里的 GitHub Actions；`GET /commits/<id>?checks=1` 可回查那些 check run 的状态。

## 测试

```bash
bash test_e2e.sh   # 本地 bare remote 全流程自测（不碰 GitHub）
```

覆盖：新分支、链式提交、新建文件、6 并发不同分支排队、坏 patch→400、noop、async+轮询、台账、队列状态。

实测也跑通了真实 GitHub（`Hana-ame/agent`）：POST 一个新建文件的 patch → push 到 `git-bridge/test-*` 分支 → GitHub contents API 与 `git ls-remote` 均确认 → 台账与 `?checks=1` 正常 → 测试后删除了该分支。

## 注意

- 服务只监听 `127.0.0.1`；要给远程 agent 用，先设 `--token`，再通过反向代理 / SSH 隧道暴露，**不要裸暴到公网**。
- patch 必须是能被 `git apply` 消化的 unified diff（`git diff` 的输出即可），不要发 `git format-patch` 的 mbox。
- 单 worker 串行：每个 job 含一次网络 push（到 GitHub ~1–3s）。高并发时会排队，`GET /queue` 看 pending 数。
