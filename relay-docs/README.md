# Agent Git Relay — 文档

本分支（`docs`）是 **git relay 系列工具** 的独立文档分支：空历史、只放文档，和代码分支互不影响。

## 三套工具一览

| 工具 | 语言 | 模式 | 适合场景 |
|------|------|------|----------|
| **gitproxy** | Go | HTTPS smart-HTTP 反代 + patch 桥 + PR/CI/issue 只读 API + 分支策略 | 常驻生产（systemd + CI/CD），当前主线 |
| **git-relay** | Node | 纯**流式** smart-HTTP 反代（无状态、不落盘） | 只要匿名 clone/push，越轻越好 |
| **git-bridge** | Node | patch-POST 桥：agent 交 diff，服务端排队 apply→commit→push | agent 不想 clone 全仓，只交改动 |

三者的**共同安全模型**：

- 仓库经本地端口 + 隧道（如 `https://wsl-8000.moonchan.xyz`）暴露给多个 agent；
- **鉴权在服务端注入**（`gh auth token` / PAT），**agent 侧零凭据**；
- 有仓库白名单，非白名单一律拒绝。

## 关系图

```
              ┌──────────────┐
 agent ──────▶│  git-relay    │──▶ github.com   （纯转发，无状态）
 agent ──────▶│  git-bridge   │──▶ 本地 clone ──▶ github.com （patch 模式，有台账）
 agent ──────▶│  gitproxy     │──▶ github.com   （反代 + patch + 元数据 API）
              └──────────────┘
                        ▲
                  systemd + CI/Release
```

## 文档索引

- [gitproxy.md](./gitproxy.md) — Go 主线工具：端点、配置、鉴权、force/delete/protect、部署、CI/Release、测试
- [git-relay.md](./git-relay.md) — Node 流式反代：白名单语法、凭据、nginx/HTTPS、参数
- [git-bridge.md](./git-bridge.md) — Node patch 桥：队列、台账、PR/CI 查询
- [ops.md](./ops.md) — 部署（systemd/隧道/权限）、CI 与 Release、策略文件格式、测试矩阵

## 一分钟选型

- 要 **agent 用标准 `git clone/push`** + 还要 PR/CI/issue 查询、分支保护、常驻服务 → 用 **gitproxy**。
- 只要 **最简匿名反代**，连本地 clone 都不想要（无状态、不落盘） → 用 **git-relay**。
- agent **不想 clone 全仓**，只把改好的 `git diff` 交上来 → 用 **git-bridge**。
