# git-relay

本地起一个 **匿名 git smart-HTTP 端点**，`git clone` / `git push` 直接打到 GitHub。
代码**不在本地落盘**——唯一的副本在 GitHub 上，relay 只是一个带白名单的管道。

```bash
git clone http://127.0.0.1:8848/Hana-ame/agent.git
git push  http://127.0.0.1:8848/Hana-ame/agent.git main
```

客户端**不需要任何凭据**。凭据只在服务端，注入上游。

Node.js 实现，**零依赖**（只用 Node 18+ 内置模块）。

---

## 为什么纯代理就够

git 的 smart-HTTP 是**无状态**的，一次 clone / push 只有两类请求：

| 步骤 | 请求 | 作用 |
|------|------|------|
| 1 | `GET  /<owner>/<repo>.git/info/refs?service=git-upload-pack` | 拿 ref 广告（capability） |
| 2 | `POST /<owner>/<repo>.git/git-upload-pack` | 发 want/have，收 packfile |
| 1' | `GET  /<owner>/<repo>.git/info/refs?service=git-receive-pack` | push 前的 ref 广告 |
| 2' | `POST /<owner>/<repo>.git/git-receive-pack` | 发 pkt-line 命令 + packfile |

服务端没有会话状态，每个请求自带全部上下文。所以把这三个端点原样流式转发给
GitHub，只在途中做两件事——**换掉 Authorization**、**校验路径在白名单里**——就能
让协议完整工作。不需要本地 bare 仓库、不需要 mirror、不需要同步任务。

> 这也是它和 `git-bridge` 的区别：`git-bridge` 是 patch 模式（POST 一个 diff，服务端
> apply + commit + push），适合 agent 交作业；`git-relay` 是协议模式，适合人（或
> CI）用正常的 `git` 命令干活。

## 快速开始

```bash
cd git-relay

# 1. 准备一个 PAT（见下节），写进环境变量
export GIT_RELAY_TOKEN=github_pat_xxxxxxxx

# 2. 起服务，白名单写文件
node server.js --port 8848 --allow-file whitelist.txt

# 3. 用
git clone http://127.0.0.1:8848/Hana-ame/agent.git
```

不带 `--token` 也能跑，但只能 clone **公开** 仓库（匿名上游）。

### 本机有全局 git 代理时必须排除 relay

如果 `git config --global http.proxy` 有值（这台机器是 `socks://localhost:10808`，
v2rayN），git 会把发往 `127.0.0.1` 的请求也塞进代理，v2rayN 转回环地址会直接断连，
现象是 `Empty reply from server`，而 relay 日志里**一条请求都没有**。

一次性排除：

```bash
git config --global http."http://127.0.0.1:8848/".proxy ""
```

这是按 URL 前缀覆盖，只影响 relay，别的仓库照旧走代理。HTTPS 就把 URL 换成
`https://127.0.0.1:8848/`。

验证过：URL 前缀对上就直连成功，故意写错端口仍然失败——因果关系是确定的。

### relay 自己访问 GitHub 慢 / 连不上

`--proxy` 让 relay 走 HTTP CONNECT 代理（v2rayN 的 http 入站、Clash 的 7890 都行）：

```bash
node server.js --port 8848 --allow-file whitelist.txt --proxy http://127.0.0.1:10809
```

只支持 `http://` 代理（CONNECT 隧道），不支持 socks——Node 内置没有 socks 客户端，
而 socks 那点协议不值得为它引一个依赖。v2rayN 同时开了 10808(socks) 和 10809(http)，
用后者。

## 白名单

这是**唯一**的访问边界，规则写在 `--allow` 或 `--allow-file` 里：

```
acme/widget        读写
acme/secret:ro     只读，能 clone 不能 push
Hana-ame/*         该 owner 下所有仓库
Hana-ame/*:ro      该 owner 下所有仓库只读
*                  全部（等于关掉白名单，别用）
```

- 最具体的规则胜出：`exact > owner/* > *`
- 大小写不敏感，`.git` 后缀可省
- 不在表里的仓库返回 **404**（不是 403），因为"没权限"本身也是信息泄露。
  要改成 403 加 `--show-unlisted`
- 表为空时**拒绝一切**（启动会打 WARNING），不会默认放开

## 凭据

relay 需要一个能 push 的 GitHub 凭据。三选一：

```bash
--token github_pat_xxx          # 直接给
--token-file /path/to/pat.txt   # 从文件读（推荐，不进 shell history）
export GIT_RELAY_TOKEN=...      # 环境变量
```

`gh` 不是必需的——GitHub 接受任意用户名配 PAT，relay 用 `x-access-token` 当用户名。
但装了 gh 的话，一条命令就能把凭据抽出来：

```bash
gh auth login                              # 一次性
gh auth token > ~/.git-relay-token         # relay --token-file 用这个
```

注意 gh 的 device-flow 最后一步是 `POST /login/oauth/access_token`，这台机器直连
GitHub 会超时，必须带上代理：

```bash
HTTPS_PROXY=http://127.0.0.1:10809 gh auth login --web
```

否则浏览器里授权成功了，gh 也会报 `read tcp ...: connection attempt failed`，
keyring 里什么都没有。

**强烈建议用 fine-grained PAT**，只授权白名单里那几个仓库：
- Repository access → Only select repositories → 勾选白名单里的
- Permissions → Contents: **Read and write**
- 其他权限全部 No access

这样即使 relay 被打穿，能造成的破坏也被 PAT 的 scope 夹住了，白名单之外还有第二道墙。

## 安全模型（重要）

**匿名 + 注入凭据 = 任何能连上这个端口的人，都能以你的 GitHub 身份 push 白名单里的仓库。**

这不是 bug，是设计。所以：

| 场景 | 做法 |
|------|------|
| 只有本机 | 默认 `--host 127.0.0.1`，别改 |
| 局域网给几台机器用 | 先加 `--basic user:pass`，再 `--host 0.0.0.0` |
| 跨公网 | **别裸暴**。套 SSH 隧道、Cloudflare Tunnel 或 nginx + TLS，并且必须 `--basic` |
| 只想给读权限 | 全部规则加 `:ro`，或直接 `--read-only` |

`--basic` 的凭据在 client 端是 URL 里的 `http://user:pass@host:port/...`，
会被 git 存进 remote url 或 credential helper。更干净的做法是让 git 用
credential helper 存，或者接受它——局域网内问题不大。

另外：

- push 的 **commit author 由客户端决定**，relay 不改写。GitHub 记录的 pusher 是
  token 的主人。所以"谁推的"在 GitHub 侧看是 PAT owner，"谁写的"看 commit author
- relay **不解包 packfile**，只审计 pkt-line 里的 ref 命令（`old new refs/heads/x`）。
  想看内容差异得靠 GitHub 侧的 CI / webhook
- 上游返回的 `WWW-Authenticate` 被剥掉——否则 git 会弹框问客户端要一个它根本
  没有、也用不上的密码

## 审计台账

每个请求一行 JSONL，落到 `--ledger`（默认 `./relay.ledger.jsonl`）：

```json
{"ts":"2026-10-31T11:24:03.512Z","method":"POST","slug":"acme/widget",
 "endpoint":"/git-receive-pack","status":200,"ms":812,"bytesIn":1043,
 "bytesOut":210,"ip":"127.0.0.1","ua":"git/2.55.0",
 "refs":[{"old":"0000…","new":"6e51…","ref":"refs/heads/feature-x","delete":false}]}
```

push 的 ref 命令是从请求体前 64 KiB 的 pkt-line 里解出来的，**不影响流式转发**。

## HTTPS

默认 HTTP。git 完全支持明文 http，本机回环场景没必要上 TLS。要上就自签：

```bash
node server.js --port 8443 --tls-generate ./tls --allow-file whitelist.txt
# 客户端信任一次：
git config --global http."https://127.0.0.1:8443/".sslCAInfo /abs/path/tls/relay-cert.pem
```

`--tls-generate` 用 openssl 生成一张自签证书（CN=localhost，SAN 带
`localhost / 127.0.0.1 / ::1`，`basicConstraints=CA:TRUE` 所以能直接当 CA 喂给
git）。已有证书就用 `--tls-cert` + `--tls-key`。

## 反向代理

**能 clone 能 push，但 nginx 默认配置下 push 会死。** 不是"可能有问题"，是必死。

### 实测：nginx 默认配置 vs 改过五行的配置

本机起了真的 nginx 1.27.4，默认配置（`proxy_buffering on`、`proxy_http_version 1.0`、
`client_max_body_size 1m`、`proxy_read_timeout 60s`）反代 relay，跑了三轮：

| 场景 | nginx 默认 | nginx 改过 |
|------|-----------|-----------|
| clone 小仓库 | ✅ 通过 | ✅ 通过 |
| **push 2MB 文件** | ❌ **413，上游什么都没收到** | ✅ 落地 |
| 限速 20KB/s、1.5MB 仓库 clone（75 秒） | ✅ 通过 | ✅ 通过 |

两条反直觉但重要的结论：

1. **`proxy_read_timeout 60s` 不是"总时长 60 秒"**。我一开始也以为是它，结果 75 秒的
   慢速 clone 在默认配置下一次都没断。它算的是**两次读操作之间的空闲时间**，只要字节
   一直在流就不触发。真正会挂的是上游（GitHub）在开始打包前长时间不吭声。
2. **默认配置下 clone 一般没事，push 一定有事**。`client_max_body_size` 默认 **1m**，
   超过就在 nginx 层直接 413，请求**根本到不了 relay**——所以 relay 台账里查不到，
   客户端只会看到一句没有上下文的 `HTTP 413`。

nginx 默认配置下的原始证据：

```
error.log: client intended to send too large chunked body: 1048440+65515 bytes,
           request: "POST /acme/widget.git/git-receive-pack"
error.log: a client request body is buffered to a temporary file
           .../temp/client_body/0000000001
git:       error: RPC failed; HTTP 413 curl 22 The requested URL returned error: 413
```

### 要改的五项

完整的可直接用配置见 [`nginx.conf.example`](nginx.conf.example)。

```nginx
location / {
    proxy_pass http://127.0.0.1:8848;
    proxy_http_version 1.1;
    proxy_set_header Connection "";

    client_max_body_size 0;        # 必须。push 可能上 GB
    proxy_request_buffering off;   # 必须。否则大 push 先落盘，撑爆 /var/lib/nginx
    proxy_buffering off;           # 强烈建议。否则首字节等到整个 packfile 收完
    proxy_read_timeout 3600s;      # 建议。默认 60s 的空闲超时对慢上游太短
    proxy_cache off;               # 必须。info/refs 被缓存 = push 时 stale ref
    gzip off;                      # 建议。packfile 已压缩
}
```

`proxy_buffering off` 其实可以不配：relay 每个响应都带 `X-Accel-Buffering: no`，
nginx 认得这个头会自己关缓冲。配上是给改过 `proxy_ignore_headers` 的人兜底。

### 子路径的坑

```
proxy_pass http://127.0.0.1:8848/;   # 结尾有斜杠：/git/acme/x.git → /acme/x.git  ✓
proxy_pass http://127.0.0.1:8848;    # 结尾没斜杠：/git/acme/x.git → 原样          ✗
```

不带斜杠时 relay 会把路径第一段 `git` 当成 owner，返回 404。路径前缀必须在
`location` 里剥掉。

### Cloudflare

CF 不是"改几个配置"的问题，是**硬限制**：

| 限制 | 数值 | 后果 |
|------|------|------|
| 请求体上限 | Free/Pro **100MB**，Business 200MB，Enterprise 500MB（可调至 5GB） | 超过直接 **413**，边缘就拒了，源站看不到 |
| 源站响应超时 | Free/Pro/Business **100 秒**固定（Enterprise 可调到 6000s） | 超时 **524** |
| 可代理端口 | HTTP: 80/8080/8880/2052/2082/2086/2095<br>HTTPS: 443/2053/2083/2087/2096/8443 | relay 默认的 **8848 不在列表里**，橙云根本连不上 |

所以：

- **端口**：要么把 relay 换成 8443（HTTPS 列表里）或 8080，要么走 Cloudflare Tunnel
  （Tunnel 不受入站端口限制，但流量仍然过 CF 边缘，下面的限制一条不少）
- **小仓库 clone / 小 push**：没问题
- **首次推一个有历史的仓库**：大概率撞 100MB，无解。要么灰云直连，要么先分几次小
  push，要么上 Enterprise
- **大仓库 clone**：CF 的 100s 是"等源站响应"，relay 是流式的、响应头立刻返回，所以
  只要 GitHub 不在开头卡 100 秒，一般能过。但没有保证，实测过若干次 524 的案例
- **缓存**：千万别给这个域名开 "Cache Everything"。`info/refs` 一旦被边缘缓存，客户端
  拿到过期 ref 列表，push 会报 `stale info` / `remote ref updated since checkout`。
  relay 透传了 GitHub 的 `cache-control: no-cache, max-age=0, must-revalidate`，
  CF 默认也不会缓存无扩展名的路径——但一个 Page Rule 就能把这个保护掀掉
- **WAF**：默认规则一般不管 GET/POST，但 "I'm Under Attack" 模式会拦 git 客户端

一句话：**Cloudflare 适合小仓库的个人用，大仓库或大 push 别指望它。**

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8848` | 监听端口 |
| `--upstream` | `https://github.com` | 上游 git 主机（GHE / Gitea / GitLab 都行） |
| `--allow` | 空 | 白名单条目，可重复 |
| `--allow-file` | 空 | 白名单文件 |
| `--token` / `--token-file` | 空 | 上游凭据 |
| `--basic` | 空 | 要求客户端 HTTP Basic，`user:pass` |
| `--read-only` | off | 全局拒绝 push |
| `--show-unlisted` | off | 非白名单返回 403 而非 404 |
| `--proxy` / `--no-proxy` | 读 `HTTPS_PROXY` | 上游走 HTTP CONNECT 代理 |
| `--tls-cert` / `--tls-key` / `--tls-generate` | 空 | HTTPS |
| `--trust-client-auth` | off | 转发客户端自己的 Authorization，而不是 `--token` |
| `--max-push-bytes` | 0 | push body 上限，0 不限 |
| `--ledger` | `./relay.ledger.jsonl` | 审计日志 |
| `--quiet` | off | 不打每请求日志 |

所有参数都有同名 `GIT_RELAY_*` 环境变量。

额外端点：

- `GET /` — 自述（白名单、用法）
- `GET /_relay/status` — 计数器（请求数 / fetch / push / 拒绝 / 错误 / 字节）

## 测试

```bash
bash test_e2e.sh
```

**完全离线**，不碰 GitHub。起一个 fake upstream（本地 bare repo + 真的
`git upload-pack` / `git receive-pack`），然后跑 23 项断言：

- clone、pull、`feature-x` 分支 push 到底有没有真的落到 upstream bare repo
- push 的文件在 upstream tree 里存在（不是只在本地 commit 了）
- 台账记到了 receive-pack 和 ref 名
- 白名单外的仓库 404 —— 包括**上游真实存在但没在白名单里**的那个
- `..` 和 `%2e%2e` 路径穿越都 404
- `:ro` 规则挡住 push 但放开 clone
- `/HEAD`、`/objects/...`、bare repo 根路径都不转发（只有三个 smart-HTTP 端点通）
- `--read-only` 全局挡 push

真实 GitHub 上跑通过：

| 场景 | 仓库 | 结果 |
|------|------|------|
| clone（直连） | `octocat/Hello-World` | ✅ |
| clone（HTTPS + 自签证书） | `octocat/Hello-World` | ✅ |
| clone（经 v2rayN HTTP 代理） | `octocat/Hello-World` | ✅ |
| **clone（12MB，经代理）** | `Hana-ame/agent` | ✅ 56s，19 个文件 |
| **push（真实分支）** | `Hana-ame/agent` | ✅ 3s，GitHub 侧 sha 与本地一致 |

push 那次的台账记录，可以对照格式：

```json
{"method":"POST","slug":"Hana-ame/agent","endpoint":"/git-receive-pack","status":200,
 "ms":1269,"bytesIn":534,"bytesOut":197,
 "refs":[{"old":"0000…","new":"5ec35d2e15fb…","ref":"refs/heads/relay-e2e","delete":false}]}
```

`old` 全零表示新建分支。测试分支用完即删，仓库没留痕迹。


## 开机自启（Windows）

```powershell
schtasks /create /tn "git-relay" /sc onlogon /rl highest ^
  /tr "\"C:\Users\lumin\.workbuddy\binaries\node\versions\22.22.2-6\node.exe\" \"C:\Users\lumin\Downloads\agent\git-relay\server.js\" --port 8848 --allow-file C:\Users\lumin\Downloads\agent\git-relay\whitelist.txt --token-file C:\Users\lumin\.git-relay-token --proxy http://127.0.0.1:10809"
```

Linux / WSL 用 systemd，`ExecStart=` 同上，配 `Restart=always`。

## 已知边界

- **不缓存**：每次 clone 都实打实从 GitHub 拉。想加速就上本地 mirror，那是另一个东西
- **不做 dumb HTTP**（`/objects/...`、`/HEAD` 被显式拒绝）—— 现代 git 不会用，放开反而是个任意文件读取的面
- **不解包 packfile**：想按文件粒度审计得在 GitHub 侧做
- **单进程无状态**：可以起多个实例挂在 nginx 后面，不需要共享存储（台账除外）
- **上游 fork 的 URL 重写**：有的 GHE 部署把仓库挂在子路径下，用 `--upstream https://ghe.corp/git` 这种带路径的形式，代码会拼在路径后面
