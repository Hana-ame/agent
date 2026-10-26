# v8 状态（2026-10-24）

> 用法看 README.md；这里只记录「做到了什么、证据、被什么卡住、恢复后先做什么」。

## 一句话

框架能力已覆盖目标：任务复用、模型缓存、格式契约、波次并行、超时/续跑、架构分解图、
provider 速度/价格观测。**实测优化被网关 530 阻塞**，offline 全链路可自检。

## 能力 → 证据

| 目标 | 实现 | 证据 |
|---|---|---|
| 公共复用 | `dag8 new` 8 模板、`scaffold --compose`、`convert` | 模板 dry-run / compose lint+dry-run |
| 不重复调查 | `scaffold --changed`（git 变更模块）、`--compose --review` | 本仓库 dogfood：3 模块 review 图 |
| 换模型不丢缓存 | 缓存键含模型；`--cache-ttl`；`dag8 cache` | 缓存测试 8 条 |
| 格式不配 | `expect`（JSON Schema 子集）+ `repair` | contracts 18 条 |
| 并行效率 | 波次调度、`--workers`、可选输入、`--max-seconds`、`--keep-going` | dag8 62 条 |
| 等不该等的 task | pi/graph 总超时、`--preflight`、`--resume`/`--resume-last` | 超时/续跑/预检测试 |
| 架构分解 | 递归 v8 子图、`scaffold --compose`、模块级评审 | scaffold 15 条 |
| 速度/价格 | `scripts/bench_pi.py`、`cost`、`stats`、`models`、`pi.fallbacks` | 基准 + 记分卡 |

## 当前证据

- 离线测试：**273 passed**（根仓库另 19 passed）；全量 277 条（含 4 live）
- **v7 本体 / v8 组织**原型：3 个短 v7 子图，v7 串行 10.83s vs v8 并行 4.0s（2.7×）；
  v8 `--cache` 第一次 7.87s、第二次 0.28s（3 子图全命中，约 28×）
- script 成为默认节点：JSON env/stdin 输入、stdout 输出（4 测试）
- 分支取材：v4 的 `join:"any"` settlement barrier 已移植（条件分支被剪后仍可汇合）
- 评审修复：远端 nvm PATH 动态解析；sorted_edges 改稳定 O(N logN)；contents_for_tag 走 JSON1 过滤（旧库回退）
- 自检：`bash scripts/smoke_offline.sh` 全绿（pytest / doctor / 模板 / compose / convert）
- 演示：`python -m dag8 demo` → `A|B`（不调模型）
- GA 账本：**77 次（65 有效 / 12 失败）**
  - 最佳：`single + deepseek-flash + terse`，15 分 / 5.5s
  - `contract`：9.46（off，n=17）→ 10.94（on，n=20），校正 +0.88；
    `chain2`：8.19（off，n=7）→ 10.81（on，n=12），交互 +3.19；绝对分 10.81 仍略低于总体基线 11.0（缓解未反转，已收敛）
  - `single/v7_only + contract` 负交互；`fanout/chain2` 负向
- 基准：pi 启动 0.28s；pi 端到端 11.4s vs v7 9.2s（固定开销约 +2.2s）
- **并行验证（2026-10-24，修正）**：2 支路 5 对 → 配对中位 1.17×（0.95–2.62）；
  4 支路 mode pi 5 对 → v7 p50 203.9s（2/5 失败）vs v8 131.4s（0 失败），有效配对仅 3 组
  （单支 67s 慢调用主导）；6 支路并发 429、限速后 120–188s。**瓶颈是 provider 并发上限
  + 单次延迟方差（7–67s），不是调度器**；免费网关下宽并行无稳定收益
- compose 真实跑：3 模块子图（36.6/92.7/48.0s）墙钟 136.5s + 汇总报告 4785 字符
- compose-review 真实跑：2 模块（28.1/104.1s）+ 汇总 111.2s，墙钟 236.8s，2961 字评审
- 缓存实测（同一 prompt）：3.86s → 0.15s（cached）；`--cache-ttl 0` → 4.13s（miss）
- 自适应并发（`--adaptive`）：429/批量超时后 worker 减半、成功 +1；`workers_used` 可观测
- 抖动退避验证（限速 mock，3 轮）：fixed p50 2.69s/7×429 vs jitter 1.65s/6×429
- 默认策略 compose 收尾：2 模块（36.6/75.8s）+ 汇总 31.7s，status ok，3593 字；
  `--preflight --json` 预检信息走 stderr，stdout 仍是纯 JSON
- 模板实机：summarize 89 字；bug_triage 输出 JSON 通过 schema（severity=high，4 键齐全）
- 递归 v8 子图实机：内层 2 并行 pi + 合并，父节点 15.6s，80 字结果（共享 rpm）
- 实机：8 并行短请求一次全成（`workers_used=[8,8]`，无误降级）；
  mock 429 集成测试 `[4,2]`
- 实机链路：keep-going → partial（bad 失败、good 保留）；修契约后 resume-last
  同一 run 只补 bad 边（3.3s），good 不重跑；explain 正确给出契约/续跑建议

## 默认策略（实测）

- 免费网关：`--workers 1~2 --rpm 3~4`；宽并行会 429/超时
- 纯 v7 迁移要速度：`convert --mode graph`（v7 直连客户端）
- provider 能力未知：`--adaptive --preflight`
- 追吞吐优先换高 RPM provider，而不是加 workers

## 网关并发实测（2026-10-24）

小请求、无重试、直接 POST `/chat/completions`：

- 1/2/3/4/6 并发：冷却期**全部 429**（连 n=1 也 429）
- 等 90s 后：n=1 仍 429；n=2 **1/2 成功**
- 结论：免费网关**没有稳定的并发阈值**，429 近似随机；并发只会放大失败率。
  `--rpm` 能避免“主动打爆”，但救不了随机 429；真正的解法是退避重试 + 换 provider。

## 当前阻塞（外部）

- `laptop-8000`：HTTP **530**（curl 直连/代理一致，源站故障），pi 快速 exit 1
- 本机 OFM 端点 `127.0.0.1:18899`：**401**（需要 key）
- 按库内纪律，**不读取凭据文件**；没有免密替代通路

## 网关恢复后待办（按顺序）

1. 复跑 `convert --mode graph` 并行对照（runs≥2，不同图规模），确认 ~3× 稳定
2. `dag8 diff A B` 对比 run；`dag8 stats` / `models` 看 429 与 p50
3. 账本可收尾：`chain2+contract` n=12，+3.19 已收敛
4. 跨 provider fallback：需要 key 或用户授权后再做
5. 用 `--mode graph` 重做 `compare_v7_v8.py` 作为默认迁移路径

## v9 / 扩展方向（尚未做）

1. ~~provider 路由~~：已实现 `route` + `--auto-model`（ok_rate − p50/120 − price_weight×price）
2. ~~成本预算~~：已实现 `--max-cost`（超了状态 over_budget，可 resume-last 续跑）
3. **分布式 workers**：把 graph/pi 节点派到远端执行，本地只做调度与落库
4. **更多节点类型**：~~`http`~~、~~`db`~~、~~`approval`~~ 已实现
5. ~~质量闸门~~：edge 级 `judge` 已实现（min_score/prompt/model）
6. ~~条件边~~：已实现 `when`（tag/pattern/equals/present/absent）

## 慢调用与质量处理（2026-10-24）

- **4 路 5 次重复**：已完成 mode pi 5 对；v8 0 失败，v7 2/5 失败（p50 203.9s vs 131.4s）。
  `--mode graph` 因子图 300s 超时，改为 pi 节点完成。
- **judge 默认阈值校准**：10/10 已评分，均值 **5.7**；默认阈值 7 只过 **3/10**
  （06_jargon 8、09_parallel 7、10_bounded 7）；structured 6、self_report 5 被拒。
  说明阈值偏严，或评审 prompt 需要按场景调整。
- 真实跨 provider：仍需 key。

处理方式：超时（节点/整体）、重试、降并发（AIMD）、fallback 都是常态路径，不把慢调用当故障。

## v9 完成度（2026-10-24）

| 能力 | 状态 | 证据 |
|---|---|---|
| 条件边 `when` | ✅ | present/absent/pattern/equals；3 测试 |
| `http` 节点 | ✅ | GET/POST/占位符/错误码/截断；5 测试 |
| `db` 节点 | ✅ | 只读 SQLite/参数绑定/JSON/截断；4 测试 |
| `approval` 人工闸门 | ✅ | **edge 级**（不是节点）：执行前检查 marker；5 测试 |
| `judge` 质量闸门 | ✅ | 输出后评分，低于阈值判失败；3 测试 |
| `--max-cost` | ✅ | over_budget + cost_estimate + explain；2 测试 |
| provider 自动路由 | ✅ | route/--auto-model；4 测试 |
| 分布式 workers | ❌ 不做 | 瓶颈是 provider RPM/并发上限，分布式不解决；增加复杂度无收益 |

**默认速度策略**：免费网关 workers 1~2 + rpm 3~4；纯 v7 迁移 `--mode graph`；
provider 能力未知用 `--adaptive --preflight`；追吞吐换 provider 而不是加 workers。

## 常用命令

```bash
python -m dag8 demo                     # 离线演示
python -m dag8 doctor                  # 探活（含 HTTP 530 识别）
bash scripts/smoke_offline.sh           # 离线全链路自检
python -m dag8 run cfg.json --preflight --rpm 2 --workers 2
python -m dag8 explain RUN_ID          # 失败诊断
python -m dag8 diff RUN_A RUN_B        # A/B 对比
python -m dag8 models                  # 价格 × 实测
```
