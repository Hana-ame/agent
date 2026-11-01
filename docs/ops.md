# 运维：部署 / CI / Release / 策略 / 测试

以 **gitproxy** 为例（当前主线）。

## 部署为 systemd 服务

| 项 | 路径 |
|----|------|
| 二进制 | `/usr/local/bin/gitproxy` |
| 配置 | `~/.config/gitproxy/repos.txt`、`branches.txt` |
| 数据（托管 clone） | `~/.local/share/gitproxy/data` |
| unit | `/etc/systemd/system/gitproxy.service`（`User=lumin`，`Restart=on-failure`） |

```ini
[Unit]
Description=gitproxy - HTTPS git relay / patch bridge to GitHub
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=lumin
Group=lumin
WorkingDirectory=/home/lumin/.local/share/gitproxy
Environment=HOME=/home/lumin
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/local/bin/gitproxy --addr :8000 \
  --data /home/lumin/.local/share/gitproxy/data --base main \
  --allow-file /home/lumin/.config/gitproxy/repos.txt \
  --protect-file /home/lumin/.config/gitproxy/branches.txt
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gitproxy
sudo systemctl status gitproxy
journalctl -u gitproxy -f
```

### 权限要点（已实测）

- 必须 **以拥有 gh/SSH 凭据的普通用户运行**（本例 `lumin`），否则读不到 `~/.config/gh` 和 `~/.ssh`。
- unit 里要显式设 `HOME`；`gh`/`git`/`ssh` 需在 `PATH` 中（`/usr/bin` 已含）。
- `/tmp` 可写（临时 patch）、`--data` 目录需对该用户可写。
- `kill -9` 后 systemd 能自动重启。

### 更新

```bash
cd gitproxy && go build -o gitproxy .
sudo cp gitproxy /usr/local/bin/gitproxy
sudo systemctl restart gitproxy
```
改配置（`repos.txt`/`branches.txt`）后同样 `systemctl restart`。

## 隧道

```
0.0.0.0:8000  ◀──  cloudflared / tunnel  ──▶  https://wsl-8000.moonchan.xyz
```
- 客户端直接用 tunnel 域名：`git clone https://wsl-8000.moonchan.xyz/Hana-ame/agent`。
- 注意 Cloudflare 对**大 pack（clone/push）** 的限制（100MB 请求体 / 100s 超时）——大仓库/大 push 可能需绕过 CF（灰云/直连/SSH）。

## CI

`.github/workflows/gitproxy-ci.yml`（路径过滤 `gitproxy/**`）：
- `go vet` → `go test`（13 个单测）→ build(linux/amd64) → 交叉编译 windows/darwin。

`.github/workflows/release.yml`（tag `gitproxy-v*` 触发）：
- 先 vet+test，再构建 **linux/amd64、linux/arm64、windows/amd64、darwin/amd64、darwin/arm64**
  5 个二进制，`gh release create` 发布并附 sha256，notes 自动生成。

本地等价：`cd gitproxy && make ci`。

## 策略文件格式

### 仓库白名单 `repos.txt`
```txt
# 一行一个；支持 owner/*
Hana-ame/agent
Hana-ame/*
other-org/*
```

### 分支策略 `branches.txt`（gitignore 风格）

```txt
# 一行一条 owner/repo/branch-glob；! = 反选(放行)；后匹配覆盖先匹配
*/*/main
Hana-ame/agent/release/**
acme/other/master
!Hana-ame/agent/release/hotfix-*
```
- 没有规则匹配→不保护；完全没配置时默认保护每个仓库的 `--base` 分支。
- 受保护分支：git push / POST /commits → 403。

## 测试矩阵（已做）

| 项 | 结果 |
|----|------|
| clone（`--branch` / `--depth 1/2/3` / `--shallow-since` / unshallow / deepen） | ✅ |
| branch / push（new、ff） / pull（跨 clone） / merge（`--no-ff` 双父） | ✅ |
| POST patch（raw / JSON） / base=分支 / base=commit sha | ✅ |
| PR（`pr=1` / `pr=title::body`） / PR 列表 / 详情 / checks | ✅ |
| CI 查询（checks / runs） / issue 列表与详情 | ✅ |
| force 非快进 → 403；删除分支/tag → 403；受保护分支 → 403 | ✅ |
| 白名单：非白名单 403；`owner/*` 通配；`--allow-file` | ✅ |
| 匿名性：干净 HOME 客户端 0 Authorization；直连 push 失败 vs 经代理成功 | ✅ |
| systemd：权限/精简环境 gh/ssh/clone、kill -9 自愈、停启 | ✅ |
| CI 在 GitHub Actions 通过；release 出 5 平台二进制 | ✅ |

## 已知边界

- force/delete/受保护分支均为 **best-effort**（解析失败会放行并记日志）；要 100% 兜底叠加 GitHub 分支保护。
- 白名单是**仓库粒度**，不限制 patch 改哪些文件。
- 只读元数据端点与 git 路径一样**不做客户端鉴权**（靠白名单 + 隧道访问控制）；`--api-token` 只保护 POST /commits。
- 公开仓库的 clone 匿名本就可读；**push 才是鉴权注入的关键证据**。
