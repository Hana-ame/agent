# git-relay（Node）

**流式、无状态** 的 git smart-HTTP 反向代理：客户端对着 relay 做标准 `git clone/push`，
relay 把 4 个 smart-HTTP 请求**流式转发**到 GitHub，并在服务端注入凭据。
本地**不存储任何仓库内容**，唯一副本在 GitHub。

```
agent ── git clone/push (HTTP) ──▶ git-relay ──(注入凭据)──▶ github.com
```

## 端点

| 方法 | 路径 |
|------|------|
| 1 | `GET  /<owner>/<repo>.git/info/refs?service=git-upload-pack` |
| 2 | `POST /<owner>/<repo>.git/git-upload-pack` |
| 1' | `GET  /<owner>/<repo>.git/info/refs?service=git-receive-pack` |
| 2' | `POST /<owner>/<repo>.git/git-receive-pack` |

## 快速开始

```bash
# 1) PAT（或从 gh 取）
gh auth token > ~/.git-relay-token

# 2) 起服务，白名单写文件
node server.js --port 8848 --allow-file whitelist.txt --token-file ~/.git-relay-token

# 3) 用
git clone http://127.0.0.1:8848/<owner>/<repo>.git
git push  http://127.0.0.1:8848/<owner>/<repo>.git <branch>
```

不带 `--token` 也能跑，但只能 clone **公开**仓库（匿名上游）。

> 本机若有全局 `git http.proxy`（如 `socks://localhost:10808`），需为 relay 单独排除：
> `git config --global http."http://127.0.0.1:8848/".proxy ""`
> 或让 relay 走上游代理：`--proxy http://127.0.0.1:10809`。

## 白名单（唯一的访问边界）

`whitelist.txt`：
```txt
# owner/repo     读写
acme/widget
# owner/repo:ro  只读（能 clone，不能 push）
acme/secret:ro
# owner/*        该 owner 下全部仓库
Hana-ame/*
# 最具体的胜出：exact > owner/* > *
*                 # 全部（危险，等于没有白名单）
```
- 匹配：`exact > owner/* > *`，大小写不敏感，`.git` 可带可不带。
- 不在表里：默认 **404**（`--show-unlisted` 时改 403）。

## 凭据

| 方式 | 说明 |
|------|------|
| `--token <pat>` | 直接给 PAT |
| `--token-file <path>` | 从文件读（推荐，不进 shell history） |
| `gh auth token` | 写入 token-file 使用 |

## 常用参数

`--port`（默认 8848）、`--host`（默认 127.0.0.1）、`--allow` / `--allow-file`、
`--token` / `--token-file`、`--proxy <url>`（上游代理）、`--basic user:pass`（客户端鉴权）、
`--read-only`、`--show-unlisted`。

## 反向代理 / HTTPS / Cloudflare

- nginx 需针对流式传输改配置（关闭响应/请求缓冲、调大超时、透传 `Git-Protocol`），
  详见 `git-relay/README.md` 的「反向代理」一节（含默认 vs 改五行实测）。
- HTTPS：relay 自身可挂 TLS 或由 nginx/隧道终止。
- Cloudflare：大 push/clone 注意 CF 的 **100MB 请求体** 与 **100s** 限制。

## 安全模型（README 摘要）

- 默认只监听 `127.0.0.1`；局域网用先加 `--basic` 再 `--host 0.0.0.0`；
- 跨公网**别裸暴**，用 SSH 隧道 / Cloudflare Tunnel / nginx+TLS，并配 `--basic`；
- 只给读权限：规则加 `:ro` 或整体 `--read-only`。
- 有审计台账（详见 README）。

## 已知边界

见 `git-relay/README.md`「已知边界」；Windows 开机自启也在其中。
