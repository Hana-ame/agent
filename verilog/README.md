# verilog：DUT + test bench

**统一工具链：Verilator（仿真）+ slang（编译检查）**。不再依赖 iverilog —— 实测
Verilator 5.055 可跑通本项目全部 7 个测试套件（含原先标为"iverilog 版"的用例）。

## 目录

| 文件 | 说明 |
|---|---|
| `counter.v` | 4-bit 同步计数器（DUT）：同步复位 / 使能 / 回绕 |
| `tb_counter.v` | counter 的 test bench（20 项断言） |
| `counter8.sv` | 8-bit 同步计数器：增加 carry 进位输出 |
| `tb_counter8_icarus.sv` | counter8 test bench（26 项断言） |
| `tb_counter8_sv.sv` | counter8 完整 SV test bench（interface/SVA/covergroup/class 随机），需 irun/VCS |
| **`sim.sh`** | **统一仿真入口**（Verilator） |
| `run_sv.sh` | EDA 服务器脚本（irun/xrun/vcs，带覆盖率） |
| `plot_waveform.py` | VCD → PNG 波形绘图（matplotlib） |

## 运行（统一入口）

```bash
bash sim.sh              # 全量回归（7 个套件）
bash sim.sh counter      # 4-bit 计数器        20 检查
bash sim.sh counter8     # 8-bit 计数器        26 检查
bash sim.sh axi          # AXI→APB→AXI 行为级  35 检查
bash sim.sh sv           # 完整 SV 事务层      234 向量
bash sim.sh gaps         # 覆盖缺口套件        285 检查
bash sim.sh dual         # 并发桥              34 检查
bash sim.sh vip          # VIP                 67 检查
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
