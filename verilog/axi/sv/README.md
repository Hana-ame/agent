# 完整 SV 版：AXI→APB→AXI（interface 端口 + SVA + class 随机）

本目录（`verilog/axi/sv/`）是与 `verilog/axi/` 等价的 **完整 SystemVerilog 版**链路，
采用标准 EDA 验证写法：

| 文件 | 角色 |
|---|---|
| `axi_lite_if.sv` | **AXI 接口定义**：五通道信号 + modport(master/slave) + clocking + SVA 断言 |
| `apb_if.sv` | **APB 接口定义**：总线信号 + modport + clocking + 阶段时序 SVA 断言（R1/R2/R3/R5） |
| `axi_apb_bridge_sv.sv` | **转换设计**：AXI4-Lite→APB 单事务串行桥（接口端口版） |
| `apb_slave_regs_sv.sv` | 慢速 APB 从机（接口端口版，wait states + 越界 SLVERR） |
| `tb_axi_apb_sv.sv` | **testbench**：class 随机激励 + scoreboard 参考模型 + covergroup 覆盖率 |

## 相比 iverilog 版的增强

| 特性 | iverilog 版（`../`） | 完整 SV 版（本目录） |
|---|---|---|
| AXI/APB 接口 | 普通端口 | **interface + modport**（master/slave） |
| 同步驱动/采样 | 手动 | **clocking block** |
| 协议检查 | task 断言 | **SVA 并发断言**（接口内自动检查） |
| 随机激励 | `$random` | **class + randomize + constraint** |
| 覆盖率 | 无 | **covergroup**（`SV_COVERAGE` 宏控制） |

## SVA 断言（协议合法性自动检查）

**AXI（`axi_lite_if.sv`）**：
- `ap_aw_valid_stable` … `ap_r_valid_stable`：5 通道 VALID 拉高后不得在握手前撤下
- `ap_w_requires_aw`：写数据必须有写事务上下文

**APB（`apb_if.sv`）**——"可以慢，但一定要合法"：
- `ap_penable_requires_psel`（R1）：PENABLE 只能在 PSEL=1 时拉高
- `ap_no_ready_in_setup`（R3）：SETUP 阶段从机不得回 PREADY
- `ap_access_stable_until_ready`（R2）：ACCESS 未完成（PREADY=0）不得退出
- `ap_xfer_done_exits`（R5）：完成拍后下一拍必须退出 ACCESS

## 编译 / 运行

```bash
# 完整 SV 编译器检查（slang，本机已验证 0 错误 0 警告）
slang --timescale=1ns/1ps --top tb_axi_apb_sv \
  -I sv -I . \
  axi_defs_pkg.sv apb_defs_pkg.sv \
  sv/axi_lite_if.sv sv/apb_if.sv \
  sv/axi_apb_bridge_sv.sv sv/apb_slave_regs_sv.sv \
  sv/tb_axi_apb_sv.sv

# 完整 SV 仿真（Verilator ≥5.055，本机已验证 202 事务全过）
verilator --binary --assert --timing --top-module tb_axi_apb_sv \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-ASCRANGE -Wno-MULTIDRIVEN \
  -I sv -I . \
  axi_defs_pkg.sv apb_defs_pkg.sv \
  sv/axi_lite_if.sv sv/apb_if.sv \
  sv/axi_apb_bridge_sv.sv sv/apb_slave_regs_sv.sv \
  sv/tb_axi_apb_sv.sv \
  -o run/tb_sv && ./run/tb_sv
# => PASS: axi->apb->axi SV testbench (202 txns, 0 errors)

# EDA 工具（irun / VCS）：断言 + 覆盖率
xrun -sv -timescale 1ns/1ps -coverage all -assert dumpoff \
  axi_defs_pkg.sv apb_defs_pkg.sv sv/*.sv
# 或
vcs -sverilog -timescale=1ns/1ps -cm line+assert axi_defs_pkg.sv apb_defs_pkg.sv sv/*.sv
```

## 工具兼容性备注（实测）

| 工具 | interface 端口 | clocking | SVA | class 随机 | covergroup |
|---|---|---|---|---|---|
| **slang 12.1** | ✅ | ✅ | ✅ | ✅ | ✅ |
| Verilator 5.055（最新） | ✅ | ✅ | ✅(子集) | ✅(constraint 需 z3) | ❌(已用 `SV_COVERAGE` 隔离) |
| Verilator 5.020（旧） | ❌ 跨文件虚接口缺陷 | ⚠️ | ✅ | ⚠️ | ❌ |
| irun / VCS | ✅ | ✅ | ✅ | ✅ | ✅ |

**实测验证（Verilator 5.055, 2026-11）**：
- `PASS: axi->apb->axi SV testbench (202 txns, 0 scoreboard errors, SVA clean)`
- 5 条 AXI VALID 稳定性断言 + 4 条 APB 阶段时序断言全部通过
- 越界访问正确回 `SLVERR`（确定性测试）

**设计/工具要点**：
- **随机化**用 `$urandom`（跨工具通用）；`rand/constraint` 保留但需 SAT 求解器
  （Verilator 需装 `z3`），slang/irun/VCS 原生支持
- **虚接口**赋值在 `initial` 里直接做（`master.vif = axi`），兼容 Verilator 静态绑定
- **握手判据**统一用 `VALID && READY`（do-while 先推一拍再查），避免
  "桥提前就绪覆盖主站 VALID 的 NBA" 死锁
- **covergroup**：Verilator 不支持，用 `SV_COVERAGE` 宏隔离；slang/irun/VCS 默认启用
- 完整 SV 链路由 slang（编译合法）+ Verilator（实际仿真）双验证