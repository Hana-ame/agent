# gitproxy

本地 Go 程序：把一个 GitHub 仓库（默认 `Hana-ame/agent`）通过 **HTTPS smart-HTTP 反代**公开，外加 **patch → commit → push → PR** 桥接，并提供 **PR / CI / issue 查询**。监听 `:8000`，配合已部署的 tunnel 对外为 `https://wsl-8000.moonchan.xyz`。

```
agent ── git clone/push (HTTPS) ──┐
agent ── POST patch (HTTPS) ──────┼──▶  gitproxy :8000  ──(注入 gh token)──▶  github.com/Hana-ame/agent
agent ── GET pulls/checks/issues ─┘        (tunnel 对外 wsl-8000.moonchan.xyz)
```

agent 端**零凭据**：不需要 gh、不需要 SSH key，鉴权全在代理侧注入。

> **提示：无特殊说明，使用 `--depth 1 --single-branch`**
> 例如 `git clone --depth 1 --single-branch https://wsl-8000.moonchan.xyz/Hana-ame/agent`。
> 只拉默认分支的 1 层历史，clone 更快、更省流量；需要完整历史 / 其他分支时再去掉这两个参数。

## 前置条件
- 本机 `gh auth login`（token 需有 `repo` scope；读私有库也需要）。
- `git`、`go`（编译用）。
- tunnel 已把 `https://wsl-8000.moonchan.xyz` → `0.0.0.0:8000`。

## 运行 / 参数

```bash
go build -o gitproxy .
./gitproxy --addr :8000 --data ./data --base main --allow Hana-ame/agent
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--addr` | `:8000` | 监听地址（tunnel 指向这里） |
| `--data` | `./data` | 托管 clone 目录（POST/force 检查用；首次 SSH clone ~1–2 分钟，之后复用） |
| `--base` | `main` | POST 新建分支的默认起点、PR 默认 base |
| `--allow` | `Hana-ame/agent` | 允许的仓库，可重复，支持 `owner/*`；值以 `@` 开头时从文件读取；非白名单一律 403 |
| `--allow-file` | 空 | 从 txt 读取白名单，**一行一个**（空行与 `#` 注释忽略），可重复 |
| `--protect` | `= --base` | 受保护分支 glob，push/commit 一律 403；`*` 单段、`**` 跨 `/`；`none` 关闭 |
| `--protect-file` | 空 | 分支策略文件（`[protect]`/`[allow]` 两段，一行一个 glob），可重复 |
| `--api-token` | 空 | 设了则 `POST /commits` 需 `Authorization: Bearer <token>`（git 路径不受限） |

也可用同名环境变量：`GIT_BRIDGE_*` 除外，这里用 `--flag`。

## 端点总表

### git（smart-HTTP 反代）
| 方法 | 路径 | 说明 |
|------|------|------|
| `GET/POST` | `/<owner>/<repo>[.git]/info/refs` | 能力广告（upload-pack / receive-pack） |
| `POST` | `/<owner>/<repo>[.git]/git-upload-pack` | clone/fetch/pull（支持 `--branch`、`--depth N`、`--shallow-since`、`--unshallow`、`--deepen`） |
| `POST` | `/<owner>/<repo>[.git]/git-receive-pack` | push（new branch / ff / delete / tag；non-ff 改写已有分支 → 403） |

### patch 桥
| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/<owner>/<repo>/commits` | body=unified diff，`?branch=&message=&base=&pr=` |

### 只读元数据（服务端 gh 拉取，agent 无需凭据）
| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/<owner>/<repo>/pulls?state=open\|closed\|merged\|all&limit=N` | PR 列表 |
| `GET` | `/<owner>/<repo>/pulls/<n>` | PR 详情 + `statusCheckRollup`（CI 汇总） |
| `GET` | `/<owner>/<repo>/pulls/<n>/checks` | PR 头提交的 CI check-runs（含 `details_url`） |
| `GET` | `/<owner>/<repo>/commits/<sha>/checks` | 某 commit 的 check-runs + combined status |
| `GET` | `/<owner>/<repo>/actions/runs?branch=&limit=N` | 最近 GitHub Actions runs |
| `GET` | `/<owner>/<repo>/issues?state=open\|closed\|all&limit=N` | issue 列表 |
| `GET` | `/<owner>/<repo>/issues/<n>` | issue 详情 + comments |

### 基础
| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 服务自述 + 端点 |
| `GET` | `/allow` | 白名单 repos + 分支策略（纯配置，无 query） |
| `GET` | `/health` | `{ok, token, allow}` |
| `GET` | `/help` | 总览：全部 endpoint + 操作说明 + **每个参数的 describe** |
| `GET` | `/help/<topic>` | 端点/操作详细帮助（clone/push/pull/merge/commits/pulls/issues/checks/runs/allow/health/force/delete/auth/whitelist/errors） |
| `GET` | `/help/<param>` | 单个启动参数 describe（addr/data/tmp-dir/base/allow/allow-file/api-token，可带或不带 `--`） |

## GET /allow（白名单 / 分支策略清单）

一次列出所有允许的仓库和分支策略，纯配置、**无 query 参数**：

```bash
B=https://wsl-8000.moonchan.xyz/Hana-ame/agent
curl -sS "$B/allow"
```

```json
{
  "ok": true,
  "repos": ["Hana-ame/agent"],
  "branch_policy": {
    "protect": ["main", "release/*"],
    "allow": [],
    "semantics": "allow overrides protect; unmatched branches are allowed (not protected)"
  },
  "base": "main",
  "notes": ["..."]
}
```

- `repos` = `--allow` 白名单（`owner/repo`，支持 `*` 通配）；
- `branch_policy.protect` / `.allow` = `--protect` / `--protect-file` 的分支 glob 规则；
- 返回的是**配置规则**，不是仓库实际分支列表；
- 只读、免鉴权（与 `/health` 一致）。

## POST /commits 细节

```bash
# A. 原始 patch（推荐）
git diff > my.patch
curl -X POST "https://wsl-8000.moonchan.xyz/Hana-ame/agent/commits?branch=feat-x&message=msg&pr=1" \
     --data-binary @my.patch

# B. JSON
curl -X POST "https://wsl-8000.moonchan.xyz/Hana-ame/agent/commits" \
  -H 'Content-Type: application/json' \
  -d '{"branch":"feat-x","message":"msg","base":"main","pr":"Title::Body","patch":"--- /dev/null\n+++ b/f\n@@ -0,0 +1 @@\n+hi\n"}'
```

Query / JSON 字段：
- `branch`（必填）—— 目标分支（不存在则新建）。
- `message`（apply 模式必填；有 `pr` 时可省）。
- `base`（可选）—— 新分支起点，**分支名或 commit sha**，默认 `--base`。
- `pr`（可选）—— `1`（标题取自 message）或 `<title>::<body>`。
- `--api-token` 开启时需 `Authorization: Bearer <token>`（或 `?token=`）。

返回（成功）：`{"ok":true,"status":"pushed","branch","sha","short","message","pr"?,"elapsedMs"}`。
其他状态：`noop`（patch 无改动）、`failed`（`detail` 带 git 报错）。
错误码：缺 branch/message/空 patch → **400**；patch 应用失败 → **400**；push 失败 → **502**。

## 查 PR / CI / issue

```bash
B=https://wsl-8000.moonchan.xyz/Hana-ame/agent
curl -sS "$B/pulls"                        # {"pulls":[{number,title,state,headRefName,baseRefName,mergeable,url,...}]}
curl -sS "$B/pulls/7"                      # {"pull":{...,"statusCheckRollup":[...]}}
curl -sS "$B/pulls/7/checks"               # {"pull":7,title,state,headSha,check_runs,combined_status}
curl -sS "$B/commits/<sha>/checks"         # {"check_runs":{total_count,check_runs:[{name,status,conclusion,details_url}]},"combined_status":{...}}
curl -sS "$B/actions/runs?limit=5&branch=main" # {"runs":[{databaseId,name,status,conclusion,headBranch,headSha,url}]}
curl -sS "$B/issues?state=all"         # {"issues":[{number,title,state,labels,author,url,comments}]}
curl -sS "$B/issues/1"                 # {"issue":{number,title,state,body,labels,comments,assignees,...}}
```

- CI 主要看 `check_runs`（`combined_status` 是旧 status API，常为 pending/空）。
- `details_url` 指到 GitHub Actions 具体 job。
- 坏 sha / 坏编号 → 400；不存在的 PR/issue → 502；非白名单 → 403。

## 分支保护策略（protect / allow txt）

`--protect-file branches.txt`（可重复）或内联 `--protect` 配置哪些分支禁止匿名 push/commit。

```txt
# branches.txt
# * = 单段，** = 跨 '/'
[protect]
main
release/*
release/**

[allow]
release/hotfix-*
```

- 语义：**先看 `[allow]`，命中则不保护；否则命中 `[protect]` 即保护；都没命中默认放行。**
- 通配：`*` 单段（不跨 `/`）、`**` 跨 `/`、`?` 单字符；也支持行内 `protect <glob>` / `allow <glob>`。
- 默认（不给参数）只保护 `--base` 分支（`main`）；`--protect none` 关闭。
- 被保护分支上 `git push` / `POST /commits` 一律 **403**（只能开 PR 由人合并）。
- 注意 git 限制：不能同时存在 `main` 和 `main/xxx` 这种同名空间，所以 `main/*` 只对真有该命名空间的仓库有意义。

## force 保护规则

receive-pack 请求体里**没有** force 标志（实测 git 2.43），所以靠**祖先关系**判断：
对「更新已有分支」（old≠0 且 new≠old）的命令，把 push 的 pack 索引进托管 clone →
`git merge-base --is-ancestor old new`：是 ff → 原样转发；不是 → **403**，分支 tip 不变。

| 操作 | 是否放行 |
|------|----------|
| 新建分支 / tag | ✅ |
| fast-forward（含普通 push） | ✅ |
| **删除分支 / tag** | ❌ 403 |
| `-f` / `--force-with-lease` 改写已有分支（non-ff） | ❌ 403 |

检查任何一步失败（托管 clone 不可用等）→ **跳过并放行**（记日志），绝不挂起 push。
要 100% 强制，再叠加 GitHub 分支保护「Restrict force pushes」。

## 鉴权 / 白名单
- 代理用 `gh auth token`（缓存 5 分钟）向 github.com 注入 `Authorization: Basic x-access-token:<token>`；客户端 Authorization 头会被丢弃。
- 白名单默认只放行 `Hana-ame/agent`；`--allow 'Hana-ame/*'` 可放宽。git / POST / 元数据端点都受同一白名单约束。
- `--api-token` 只保护 POST，不影响 git clone/push。

## push 去哪
- **agent `git push`**：agent → 代理 → `https://github.com/<owner>/<repo>.git`（注入 token），落到 GitHub `refs/heads/<branch>`。
- **`POST /commits`**：服务端在托管 clone `data/<owner>-<repo>`（origin 为 `git@github.com:...` SSH）里 apply+commit，再 `git push origin HEAD:refs/heads/<branch>`，同样落到 GitHub。
- 代理本身不持久化 push，只做转发/改写；本地只有 `data/` 里的托管 clone 对象副本。

## CI

- 单元测试 `main_test.go`：纯函数、**无网络**（glob 匹配、分支策略文件解析、pkt-line 解析、路由/白名单解析等 13 个用例）。
- 本地一键：`make ci`（`go vet` + `go test` + linux/windows/darwin 交叉编译 + build）。
- GitHub Actions：`gitproxy/.github/workflows/ci.yml`（vet / test / build / 交叉编译）。
  - **gitproxy 作为独立仓库**：把该文件放到仓库根的 `.github/workflows/ci.yml` 即可。
  - **嵌在别的仓库的子目录**：移到仓库根 `.github/workflows/gitproxy-ci.yml`，加 `defaults.run.working-directory: gitproxy`，并把 `gitproxy/` 提交进该仓库。

## 运维 / 已知限制
- clone 过代理偶发 `invalid index-pack output` 已通过**强制后端 HTTP/1.1** 解决；仍偶有网络抖动，重试即可。
- 托管 clone 首次慢（~1–2 分钟）；删除 `data/` 会触发重新 clone。
- 只读元数据端点无鉴权（与 git 路径一致）；若担心，靠白名单 + tunnel 访问控制。
- `POST /commits` 不校验 patch 是否只动白名单文件；白名单是仓库粒度。