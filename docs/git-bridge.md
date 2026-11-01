# git-bridge（Node）

**patch-POST 桥**：agent 不 clone 仓库，直接把 `git diff` POST 过来；服务端在
**单 worker 队列**里依次 `fetch → apply → commit → push`，并把每次结果写进**持久化台账**。

```
agent ── POST 一个 diff ──▶ git-bridge（FIFO 单 worker）
                              fetch → checkout -B → apply → commit → push
                              └── 台账 ledger.jsonl（可查状态 / 驱动 CI）
```

- 零 npm 依赖（Node 内置模块）。
- 串行队列：任意时刻只有一个 git 操作，**无并发竞争**。
- push/fetch 复用本机 `gh` 凭据（`gh auth setup-git`）。

## 启动

```bash
# 指向已有本地 clone
node server.js --repo /path/to/repo --base main

# 或让服务端自己 clone
node server.js --repo ./myrepo --repo-url git@github.com:you/repo.git --base main
```

参数：`--repo`、`--repo-url`、`--base`、`--host/--port`（默认 8765）、`--token`、`--author`、`--ledger`、`--wait-timeout`。

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/commit?branch=&message=&base=&pr=` | body = unified diff；入队 apply+commit+push |
| GET | `/commits?branch=&status=&limit=` | 台账列表（最新在前） |
| GET | `/commits/<id>` | 单条详情；`?checks=1` 附带该 commit 的 GitHub CI |
| GET | `/queue` | 队列深度 / worker 状态 |
| GET | `/health` | 仓库 HEAD、origin、gh 登录状态 |
| GET | `/branches` | 远端分支 |
| GET | `/` | 自述 |

POST 细节：
- body 可为 raw diff 或 JSON `{branch,message,patch,base,pr}`；
- `async=1` 立即返回 `202 + id`（再轮询 `/commits/<id>`）；默认同步等待（`--wait-timeout` 超时转 202）；
- `notify=<url>` 完成后 POST 结果给下游（驱动 CI）；
- 返回 `{ok,status: pushed|noop|failed, sha, pr?, elapsedMs}`。

## 与 gitproxy 的区别

| | git-bridge | gitproxy |
|---|---|---|
| agent 怎么提交 | POST 一个 diff | 标准 git push 或 POST diff |
| clone | 不提供 | 反代 clone |
| 队列 | 单 worker 串行 | 并发转发 |
| 台账 / PR/CI 查询 | 有台账 + `?checks=1` | 有 pulls/checks/runs/issues API |
| 分支保护 / force / 删除防护 | 无 | 有 |

## 测试

`test_e2e.sh`（本地 bare remote，不碰 GitHub）：新分支、链式提交、新建文件、
6 并发分支排队、坏 patch→400、noop、async+轮询、台账、队列。
