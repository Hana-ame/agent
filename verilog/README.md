# verilog：DUT + test bench

**统一工具链：Verilator（仿真 + 覆盖率）+ slang（编译检查）**。不再依赖 iverilog。

当前规模：**25 个测试套件 · 约 1400 项检查 · RTL 有效覆盖率 100% · 变异得分 100%（29/29）**

## 目录

| 文件 | 说明 |
|---|---|
| `counter.v` | 4-bit 同步计数器（DUT）：同步复位 / 使能 / 回绕 |
| `tb_counter.v` | counter 的 test bench（20 项断言） |
| `counter8.sv` | 8-bit 同步计数器：增加 carry 进位输出 |
| `tb_counter8_icarus.sv` | counter8 test bench（26 项断言） |
| `tb_counter8_sv.sv` | counter8 完整 SV test bench（interface/SVA/covergroup/class 随机），Verilator 可跑 |
| **`sim.sh`** | **统一仿真入口**（Verilator） |
| **`ci.sh`** | **质量门 CI**（9 关定义，默认跑 6 关） |
| **`audit.sh`** | **验证工具自审计**（8 项故障注入，确认工具会报错） |
| **`repro.sh`** | **复现性验证**（干净 clone 后能否跑通） |
| **`stability.py`** | **稳定性检查**（套件重复运行检测偶发失败） |
| **`axi/docs_check.py`** | **文档一致性**（模块清单/测试项数/废弃工具残留/HTML 结构） |
| `run_sv.sh` | EDA 服务器脚本（irun/xrun/vcs，带覆盖率） |
| `plot_waveform.py` | VCD → PNG 波形绘图（matplotlib） |
| `axi/` | AXI→APB→AXI 设计、验证、文档（见下） |

### axi/ 子目录

| 文件 | 说明 |
|---|---|
| `axi_defs_pkg.sv` / `apb_defs_pkg.sv` | AXI4-Lite / APB 协议类型定义 |
| `sv/axi_lite_if.sv` / `sv/apb_if.sv` | 接口 + modport + clocking + SVA |
| `sv/axi_apb_bridge_sv.sv` | 串行桥（单事务 FSM，11 状态） |
| `sv/axi_apb_bridge_dual_sv.sv` | 并发桥（读写独立 FSM + 仲裁，13 状态） |
| `sv/apb_slave_regs_sv.sv` | APB 从机（纯存储 + 可配 wait states） |
| `sv/apb_slave_regfile_sv.sv` | 带属性从机（RO/RW/W1C/RSVD/W1S/RC） |
| `sv/axi_txn_host.sv` | 事务适配器（txn_t → AXI 时序，测试复用） |
| `sv/axi_apb_monitor.sv` | **独立协议监视器**（被动只读，自动检查 AXI/APB 合法性） |
| `sv/bus_txn_pkg.sv` | 事务结构体（数据位宽由 `TXN_DW` 宏配置，默认 32） |
| `regfile_spec.json` | 寄存器规格（唯一事实来源，含位域） |
| `gen_regfile.py` | spec → RTL / TB / 手册 生成器 |
| `gen_fsm_html.py` | 源码 → FSM 状态转移图 |
| `gen_interface_map.py` | 源码 → 接口逻辑映射 |
| `gen_test_plan.py` | 源码 → 测试方案 |
| `gen_coverage_html.py` | 覆盖率注解 → 可视化报告 |
| `gen_portal.py` | 文档门户（统计实时提取） |
| `mutate.py` | 变异测试框架 |

## 运行（统一入口）

### 测试套件（11 个）

```bash
bash sim.sh              # 全量回归
bash sim.sh counter      # 4-bit 计数器              20 检查
bash sim.sh counter8     # 8-bit 计数器              26 检查
bash sim.sh axi          # AXI→APB→AXI 行为级        35 检查
bash sim.sh sv           # 完整 SV 事务层            234 向量
bash sim.sh gaps         # 覆盖缺口套件              285 检查
bash sim.sh dual         # 并发桥（含反压/错误传播）  58 检查
bash sim.sh stress       # 并发仲裁压力              505 检查
bash sim.sh covgaps      # 覆盖率缺口专项             17 检查
bash sim.sh monitor      # 协议监视器自测（注入违规）  2 违规捕获
bash sim.sh regfile      # 寄存器属性（手写）         44 检查
bash sim.sh regfilegen   # 寄存器属性（spec 生成）    70 检查
bash sim.sh vip          # VIP                        67 检查
```

### 质量门 CI

```bash
bash ci.sh              # 默认 4 关：编译检查 + 快速回归 + 全量回归 + 文档（~35s）
bash ci.sh --full       # 全部 7 关（含参数扫掠/覆盖率/变异，~10min）
bash ci.sh --stage 5    # 只跑第 5 关
bash ci.sh --list       # 列出所有关卡
```

| 关卡 | 内容 | 阈值 | 典型耗时 |
|---|---|---|---|
| G1 | slang 编译检查（4 组） | 全部通过 | <1s |
| G2 | 快速回归（4 套件） | 全部通过 | ~1s |
| G3 | 全量回归（11 套件） | 全部通过 | ~33s |
| G4 | 参数扫掠（9 种子 × 9 wait） | 18/18 | ~5min |
| G5 | RTL 有效覆盖率 | ≥ 90% | ~50s |
| G6 | 变异得分 | ≥ 80% | ~250s |
| G7 | 文档生成（6 份 HTML） | 全部生成 | ~1s |

任一关失败立即停止并返回 exit code 1，可直接用于 CI 集成。

### 复现性验证

```bash
bash repro.sh            # 完整（clone + 全量回归，约 5 分钟）
bash repro.sh --quick    # 快速（clone + 4 套件，约 40 秒）
```

4 步检查：干净 clone → 文件完整性 → 生成物一致性 → 全量回归。

> 为什么需要这个：本项目曾出现「本地能跑、干净 clone 跑不通」——
> `axi/run` 目录未自动创建、手改的生成物被生成器覆盖。
> 这类问题在本地开发时完全不可见（本地已有残留状态），
> 只有从零 clone 才能暴露。

### 验证工具自审计

```bash
bash audit.sh            # 6 项故障注入审计
bash audit.sh --list     # 列出审计项
```

对每条验证链路做「注入缺陷 → 确认报错 → 恢复」的闭环：

| 审计 | 内容 | 期望 |
|---|---|---|
| A1 | 单套件注入缺陷 | exit≠0 且输出 FAIL |
| A2 | 全量回归注入缺陷 | 中断且 exit≠0 |
| A3 | CI G3 注入缺陷 | 报告失败 |
| A4 | 变异框架残留检测 | exit=4 |
| A5 | 协议监视器自测 | 能抓违规 |
| A6 | 覆盖率阈值拒绝 | 低覆盖被拒 |

目的：确认工具在失败时**不会静默通过**。

### 工具目标

```bash
bash sim.sh coverage     # 覆盖率报告 + HTML
bash sim.sh mutate       # 变异测试（验证测试有效性）
bash sim.sh sweep        # 参数扫掠（9 种子 × 9 wait）
bash sim.sh docs         # 生成全部 HTML 文档
bash sim.sh wave         # 完整 SV + VCD 波形
```

参数化回归：

```bash
WAIT=0  bash sim.sh gaps    # 零等待
WAIT=16 bash sim.sh gaps    # 长等待
SEED=123 bash sim.sh gaps   # 多 seed
```

实测全量回归输出：

```
== sim: counter   ==  PASS: all 20 checks passed
== sim: counter8  ==  PASS: counter8 all checks passed
== sim: axi       ==  PASS: axi->apb->axi all 35 checks passed (monitor clean)
== sim: sv        ==  PASS: axi->apb->axi transaction-layer testbench (234 txns)
== sim: gaps      ==  PASS: gaps suite (WAIT=2 SEED=0)
== sim: dual      ==  PASS: dual-path bridge concurrency
== sim: vip       ==  VIP PASS: 67 checks, 0 errors
```

## 不再使用 iverilog 的理由（实测）

| 能力 | iverilog 12 | Verilator 5.055 |
|---|---|---|
| Verilog-2001 仿真 | ✅ | ✅ |
| `interface` 作为模块端口 | ❌ | ✅ |
| `clocking` block | ❌ | ✅ |
| SVA `assert property` | ❌ | ✅（子集） |
| `class` / `$urandom(seed)` | ❌ | ✅ |
| VCD 波形（`--trace`） | ✅ | ✅ |
| `-G` 参数覆盖（`WAIT`/`SEED`） | ❌ | ✅ |
| `covergroup` | ❌ | ❌（仍需 EDA 工具） |
| UVM 运行时仿真 | ❌ | ❌（需 EDA 工具；编译可用 slang 验证） |

**本项目所有用例均已在 Verilator 下实测通过**，iverilog 无独有能力，故移除。
仅保留两个例外需要 EDA 工具：`covergroup` 覆盖率收集、UVM 运行时仿真。

## 工具分工

| 工具 | 角色 |
|---|---|
| **slang** | 编译检查（仿真前）：完整 SV 语法/语义，含 UVM 库校验 |
| **Verilator** | 仿真：全部 7 个套件 |
| irun / VCS | 覆盖率、UVM 运行时、时序仿真（服务器） |

## 激励的起始 / 结束形式（TB 生命周期）

每个 test bench 按 **startup → 阶段测试 → drain → finalize** 组织激励：

```
┌─ 起始激励 startup ─────────────────────────────────────────────
│  上电复位序列（power-on reset）：
│    rst_n=0 保持 2 周期 → 检查归零  (DUT 全部触发器确定态)
│    rst_n=1 再观察 2 周期 → 确认稳定后再进入测试
├─ 阶段测试（确定性 + 随机，见各 TB）
├─ 结束激励 1 drain ─────────────────────────────────────────────
│  激励回到安全态 → 跑 3 周期冲洗边沿 → 末拍保持检查
├─ 结束激励 2 finalize ──────────────────────────────────────────
│  打印 SUMMARY（错误数 / 周期数）
│  0 错误 → $finish      （退出码 0）
│  有错误 → $fatal(1)    （退出码 1，CI 可捕获）
└────────────────────────────────────────────────────────────────
```

**watchdog 防挂死**（可选）：

```bash
bash sim.sh sv   # 若需 watchdog，在 TB 中用 `ifdef TB_WATCHDOG 包裹超时 $fatal
```
