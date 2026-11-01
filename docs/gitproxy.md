# gitproxy（Go）

本地 Go 程序：把一个 GitHub 仓库通过 **HTTPS smart-HTTP 反代**公开给多个 agent，并提供
**patch → commit → push → PR** 桥接、**PR/CI/issue 只读查询**、**force/删除/受保护分支防护**。
监听一个端口，常配隧道对外（例：`:8000` ←夜 `https://wsl-8000.moonchan.xyz`）。

- 零第三方依赖（Go 标准库）。
- agent 侧**零凭据**：代理用 `gh auth token` 向上游 github.com 注入 `Authorization: Basic x-access-token:<token>`。

## 端点

### git（smart-HTTP 反代，标准协议）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET/POST | `/<owner>/<repo>[.git]/{info/refs,git-upload-pack,git-receive-pack}` | clone / fetch / pull / push |

原生支持 `--branch`、`--depth N`、`--shallow-since`、`--unshallow`、`--deepen`。

### patch 桥
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/<owner>/<repo>/commits` | body = unified diff；`?branch=&message=&base=&pr=` |

- `base`：新分支起点，**分支名或 commit sha**（默认 `--base`）。
- `pr`：`1`（标题取 message）或 `<title>::<body>`。
- 返回：`{ok, status: pushed|noop|failed, sha, short, pr?, elapsedMs}`。
- 错误码：缺参/空 patch/patch 失败 → **400**；受保护分支 → **403**；push 失败 → **502**。

### 分支 API
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/<owner>/<repo>/branches?limit=N` | 分支列表 |
| GET | `/<owner>/<repo>/branches/<name>` | 分支详情（指向的 commit） |
| POST | `/<owner>/<repo>/branches` | 创建分支：`name=`、`from=`（分支/commit sha/tag，默认 base）；支持 JSON |

- 已存在 → 409；`from` 解析不到 → 400；上游失败 → 502。
- `git push` 新分支、`POST /commits` 到不存在分支也会建分支。

### 只读元数据（服务端 `gh` 拉取，agent 无需凭据）
| 方法 | 路径 |
|------|------|
| GET | `/<owner>/<repo>/pulls?state=&limit=` |
| GET | `/<owner>/<repo>/pulls/<n>`（含 `statusCheckRollup`） |
| GET | `/<owner>/<repo>/pulls/<n>/checks` |
| GET | `/<owner>/<repo>/commits/<sha>/checks` |
| GET | `/<owner>/<repo>/actions/runs?branch=&limit=` |
| GET | `/<owner>/<repo>/issues?state=&limit=` |
| GET | `/<owner>/<repo>/issues/<n>`（含 comments） |

### 帮助 / 健康
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/help` | 总览：端点 + 参数 describe |
| GET | `/help/<topic>` | 主题：clone/push/pull/merge/commits/pulls/issues/checks/runs/health/force/delete/auth/whitelist/policy/errors |
| GET | `/help/<param>` | 单个启动参数（可带/不带 `--`） |
| GET | `/health` | `{ok, token, allow}` |

> 没有 **PR merge** 端点（设计如此）：PR 只能由人合并。

## 配置参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--addr` | `:8000` | 监听地址（隧道指向这里） |
| `--data` | `./data` | 托管 clone 目录（POST/force 检查用；首次 SSH clone 后复用） |
| `--tmp-dir` | OS 临时目录 | 临时 patch 目录（Linux `/tmp`，Windows `%TEMP%`），用完即删 |
| `--base` | `main` | 新分支默认起点 / PR 默认 base |
| `--allow` | `Hana-ame/agent` | 仓库白名单，可重复；支持 `owner/*`、`@file` |
| `--allow-file` | 空 | 白名单 txt，一行一个（`#` 注释、空行忽略） |
| `--protect` | `= --base` | 受保护分支 glob，`*` 单段、`**` 跨 `/`；`none` 关闭 |
| `--protect-file` | 空 | 分支策略 txt（`[protect]`/`[allow]`） |
| `--api-token` | 空 | 设置后 POST /commits 需 `Bearer`（git 路径不受限） |

环境：本机 `gh auth login`（需 `repo` scope）、`git`。

## 鉴权模型

1. 启动/请求时取 `gh auth token`（缓存 5 分钟）。
2. 转发到 github.com 时注入 `Authorization: Basic base64("x-access-token:"+token)`。
3. **客户端发来的 `Authorization` 被丢弃**；git clone/push 端点不要求客户端鉴权。
4. 结果：agent 不需要 gh、不需要 SSH key、不需要 token。

## 防护规则

### force（非快进改写）
- 对「更新已有分支」（old≠0 且 new≠old）的 receive-pack 命令，把 push 的 pack 索引进托管 clone，
  `git merge-base --is-ancestor old new` 判断：是 ff → 放行；不是 → **403**。
- 新建分支 / 普通 ff push → 放行。
- 任何一步失败（clone 不可用等）→ 跳过并放行（记日志），**绝不挂起 push**（best-effort）。

### 删除
- 经代理删除分支 / tag（`git push --delete` / `:refs/tags/x`）→ **403**。

### 受保护分支（gitignore 风格，按 owner/repo/branch）

一行一条 `owner/repo/branch-glob`，`!` 前缀=反选（放行），**后匹配覆盖先匹配**。
没规则匹配→不保护；完全没配置时默认保护每个仓库的 `--base` 分支；`--protect none` 关闭。

```txt
*/*/main
Hana-ame/agent/release/*
Hana-ame/agent/release/**
acme/other/master
!Hana-ame/agent/release/hotfix-*
```

- owner/repo 支持 `*`/`?`；branch 支持 `*` 单段、`**` 跨 `/`、`?` 单字符。
- 顺序生效（gitignore 语义），非法行按 file:line 报错。
- 被保护分支：git push / POST /commits → **403**。

## 仓库白名单

```txt
# repos.txt（--allow-file）
Hana-ame/agent
Hana-ame/*
other-org/*
```
- 非白名单：clone/push/POST/查询一律 **403**。
- 默认只放行 `Hana-ame/agent`。

## 数据与临时文件

- **持久**：`--data`（默认 `./data`）里的托管 clone（首次 SSH clone ~1–2 分钟，之后复用；
  其 `.git/objects` 会累积，可 `git gc` 或删掉重 clone）。
- **临时**：`POST /commits` 的 patch 写到 `--tmp-dir`，`git apply` 后**立即删除**。
- 无数据库、无其他状态文件；日志走 stdout/stderr。

## 构建与运行

```bash
go build -o gitproxy .
./gitproxy --addr :8000 --data ./data --base main \
  --allow-file ./repos.txt --protect-file ./branches.txt
```

## 测试
- `make ci`：`go vet` + `go test` + linux/windows/darwin 交叉编译 + build。
- `main_test.go`：13 个纯函数单测（glob/policy/pkt-line/路由/白名单等，无网络）。
