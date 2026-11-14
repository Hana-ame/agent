# 完整 SV 版：AXI→APB→AXI（三层抽象 + 事务结构体 + 向量闭环）

本目录（`verilog/axi/sv/`）是与 `verilog/axi/` 等价的 **完整 SystemVerilog 版**链路。

## 三层抽象：定义简单到就是结构体

```
┌──────────────────────────────────────────────────────────────┐
│ 第 3 层  bus_txn_pkg.sv —— 只有两个 struct，一眼读完          │
│   txn_t { write, addr, data, pause, same, exp_err }          │
│   rsp_t { data, resp, idle }                                 │
│   testbench 只写：host.execute(t, r);   ← 零协议细节           │
└───────────────────────┬──────────────────────────────────────┘
                        ▼  axi_txn_host 展开
┌──────────────────────────────────────────────────────────────┐
│ 第 2 层  axi_txn_host.sv —— 事务适配器                        │
│   txn_t → AXI 握手时序（AW/W 同发或分开发、AR/R 读）           │
│   pause → 事务后插入 N 拍空闲（总线全 VALID=0）                │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ 第 1 层  协议细节                                             │
│   axi_lite_if (modport/clocking/5 条 SVA)                     │
│   axi_apb_bridge_sv → apb_if (4 条阶段 SVA) → slave           │
└──────────────────────────────────────────────────────────────┘
```

**关键收益**：`tb_axi_apb_sv.sv` 里**不出现任何 AXI 信号名**（无 AWVALID/WREADY/clocking），
测试场景完全由 `txn_t` 结构体描述。

| 文件 | 角色 |
|---|---|
| `bus_txn_pkg.sv` | **第 3 层**：`txn_t` / `rsp_t` 纯数据结构体（含 `pause` 控制字段） |
| `axi_txn_host.sv` | **第 2 层**：事务适配器，`txn_t` ↔ AXI 握手时序 |
| `axi_lite_if.sv` | **第 1 层**：AXI 接口（modport + clocking + 5 条 SVA 断言） |
| `apb_if.sv` | **第 1 层**：APB 接口（modport + clocking + 4 条阶段断言 R1/R2/R3/R5） |
| `axi_apb_bridge_sv.sv` | **转换设计**：AXI4-Lite→APB 单事务串行桥（11 态 FSM） |
| `apb_slave_regs_sv.sv` | 慢速 APB 从机（wait states + 越界 SLVERR） |
| `tb_axi_apb_sv.sv` | **testbench**：只用 `txn_t`/`rsp_t` 的向量闭环 |

## pause 字段（事务间暂停）

`txn_t.pause` 表示该事务完成后插入多少拍总线空闲：

```systemverilog
txn_t t = txn(1'b1, 8'h00, 32'hDEAD_BEEF, 8'd3, 1'b0, 1'b0);
//                        ↑addr  ↑data        ↑pause=3 拍空闲
host.execute(t, r);
// r.idle == 3    ← 适配层回读实际空闲拍数，可用于断言
```

- 上层只声明「要暂停几拍」，不知道 AXI 如何表达空闲
- 第 2 层把 pause 展开为：拉低全部 VALID + 等待 N 拍
- `rsp_t.idle` 回传实际注入的空闲拍，testbench 逐条核对（本机实测 164 项 pause 检查全过）
- 覆盖 `pause=0~3` 随机分布 + 确定性向量，共注入 311 个空闲拍

## 向量化测试闭环（tb_axi_apb_sv.sv）

```
① 数据向量集 stim_vecs[$]：txn_t 队列（回环 32 + 随机 200 + 越界 2）
        ▼
② 激励向量 host.execute(t, r)：展开成总线时序（含 pause 空闲）
        ▼
③ 数据向量 check_rsp(t, r)：r vs 期望（ref_mem 镜像）比对
   · 写向量 → 更新 ref_mem
   · 读向量 → 与 ref_mem 比对
   · pause  → r.idle 必须等于 t.pause
```
└───────────────┬───────────────────────────────────────────────┘
                ▼
┌───────────────────────────────────────────────────────────────┐
│ ③ 数据向量比对  check_rsp(t, r) 期望 vs 实际                    │
│    写向量：r.resp==OKAY → ref_mem[addr]=t.data（更新期望）       │
│    读向量：r.data !== ref_mem[addr] → FAIL                      │
│    越界  ：r.resp !== SLVERR → FAIL                              │
│    pause ：r.idle !== t.pause → FAIL                            │
└───────────────────────────────────────────────────────────────┘
      ▲              反向闭环
      └── ③ 更新的 ref_mem 就是 ① 后续读向量的比对基准
```

**闭环语义**：写数据写入期望向量集 → 读向量以它为基准比对 → DUT 往返
（AXI→APB→寄存器→APB→AXI）数据一致即 PASS。仿真结果快照 `ref_mem`
使读写一致性可审阅。

## 相比混合式 TB 的增强

| 特性 | 旧版（随机流式） | 新版（向量化闭环） |
|---|---|---|
| 激励组织 | 随机逐事务 | **数据向量集**（含期望响应） |
| 期望管理 | 散在 scoreboard | **参考镜像 ref_mem**（写更新/读比对） |
| 执行流 | 生成即驱动 | **向量驱动 → 捕获 → 比对**三段显式 |
| 可追溯 | 事件流 | 每条向量可审查（结构体字段） |
| 贯通性 | 读写交错验证 | **写→读回环**显式验证数据往返 |

## 相比 iverilog 版的增强

| 特性 | iverilog 版（`../`） | 完整 SV 版（本目录） |
|---|---|---|
| 抽象层次 | 直接操作端口信号 | **三层抽象**（txn_t 结构体 → 适配器 → 协议） |
| AXI/APB 接口 | 普通端口 | **interface + modport**（master/slave） |
| 同步驱动/采样 | 手动 | **clocking block** |
| 协议检查 | task 断言 | **SVA 并发断言**（接口内自动检查） |
| 随机激励 | `$random` | **`$urandom` + class 可选** |
| 事务暂停 | 无 | **`txn_t.pause` 字段**（事务间空闲拍） |
| 覆盖率 | 无 | **covergroup**（`SV_COVERAGE` 宏控制） |

## SVA 断言（协议合法性自动检查）

**AXI（`axi_lite_if.sv`）**：
- `ap_aw_valid_stable` … `ap_r_valid_stable`：5 通道 VALID 拉高且未握手时，下一拍必须保持

**APB（`apb_if.sv`）**——"可以慢，但一定要合法"：
- `ap_penable_requires_psel`（R1）：PENABLE 只能在 PSEL=1 时拉高
- `ap_no_ready_in_setup`（R3）：SETUP 阶段从机不得回 PREADY
- `ap_access_stable_until_ready`（R2）：ACCESS 未完成（PREADY=0）不得退出
- `ap_xfer_done_exits`（R5）：完成拍后下一拍必须退出 ACCESS
- `cov_xfer_done`：cover property 记录传输完成拍

## 编译 / 运行

```bash
# 完整 SV 编译器检查（slang，本机已验证 0 错误 0 警告）
slang --timescale=1ns/1ps --top tb_axi_apb_sv \
  -I sv -I . \
  axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
  sv/axi_lite_if.sv sv/apb_if.sv \
  sv/axi_apb_bridge_sv.sv sv/apb_slave_regs_sv.sv \
  sv/axi_txn_host.sv sv/tb_axi_apb_sv.sv

# 完整 SV 仿真（Verilator ≥5.055，本机已验证 234 事务全过）
verilator --binary --assert --timing --top-module tb_axi_apb_sv \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-ASCRANGE -Wno-MULTIDRIVEN \
  -I sv -I . \
  axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
  sv/axi_lite_if.sv sv/apb_if.sv \
  sv/axi_apb_bridge_sv.sv sv/apb_slave_regs_sv.sv \
  sv/axi_txn_host.sv sv/tb_axi_apb_sv.sv \
  -o run/tb_sv && ./run/tb_sv
# => PASS: axi->apb->axi transaction-layer testbench (234 txns)

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
- `PASS: axi->apb->axi transaction-layer testbench (234 txns)`
- `== 3 data compare: 234 checks, 164 pause checks, 0 errors ==`
- `-- SUMMARY: 234 txns, 311 pause cycles injected --`
- 5 条 AXI VALID 稳定性断言 + 4 条 APB 阶段时序断言全部通过
- 越界访问正确回 `SLVERR`；pause 字段注入的 311 个空闲拍全部可观测回读

**设计/工具要点**：
- **随机化**用 `$urandom`（跨工具通用）；`rand/constraint` 保留但需 SAT 求解器
  （Verilator 需装 `z3`），slang/irun/VCS 原生支持
- **虚接口**赋值在 `initial` 里直接做（`master.vif = axi`），兼容 Verilator 静态绑定
- **握手判据**统一用 `VALID && READY`（do-while 先推一拍再查），避免
  "桥提前就绪覆盖主站 VALID 的 NBA" 死锁
- **covergroup**：Verilator 不支持，用 `SV_COVERAGE` 宏隔离；slang/irun/VCS 默认启用
- 完整 SV 链路由 slang（编译合法）+ Verilator（实际仿真）双验证