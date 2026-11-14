# AXI4-Lite VIP：可直接接入的验证模块

一个模块同时包含**打激励**和**验证激励**，通过 interface 接入任意 AXI4-Lite DUT。

## 两种形态

| 形态 | 文件 | 依赖 | 状态 |
|---|---|---|---|
| **非 UVM VIP** | `axi_vip_pkg.sv` + `axi_lite_vip.sv` | 无（纯 SV） | ✅ 本地实测 PASS |
| **UVM 配套** | `axi_vip_uvm_pkg.sv` + `tb_uvm_top.sv` | UVM 库 + EDA 仿真器 | 骨架就绪，**本机无法验证** |

### 本机 UVM 可行性（已核查）

- ✅ 系统里**有 UVM 库源码**：`modeltech_10.2c/verilog_src/uvm-1.1b|c|d`、`questa_uvm_pkg-1.2`
- ❌ 但**没有任何 EDA 仿真器可执行文件**：`vsim`/`vlog`/`xrun`/`irun`/`vcs` 均不在 PATH，
  该 modeltech 目录只有库文件没有二进制
- ❌ Verilator 5.055 **不支持 UVM**（无 `uvm_pkg` 依赖的 factory/phase 机制）

因此：UVM 版本按标准结构写好并做了语法级梳理，但**未在真机仿真验证**，
需在装有 irun/VCS/xsim 的机器上跑 `tb_uvm_top.sv` 确认。
本地真实验证的是非 UVM VIP（67 检查 PASS）。

## 三行接入

```systemverilog
// 1. 实例化（VIP 占 AXI master 角色，DUT 接 slave 侧）
axi_lite_vip #(.AW(8), .DW(32)) vip (
    .ACLK(clk), .ARESETn(rst_n), .axi(axi_if)
);

// 2. 配置 + 打激励
vip.setup(.limit(64), .vb(0), .word_ma(1));
automatic axi_sequence seq = new();
seq.add_loopback(8'h00, 32'hDEAD_BEEF, 8'd1);   // 写后读回
seq.add_random(60, 64);                          // 随机
vip.run_sequence(seq);
vip.execute_all();

// 3. 验证结果
if (!vip.report()) $fatal(1, "VIP FAILED");
```

## 内部结构

```
┌───────────────────── axi_lite_vip ─────────────────────┐
│                                                         │
│  ① axi_sequence    生成事务队列（loopback/random/自定义）│
│        ↓                                                │
│  ② driver          展开成 AXI 握手（AW/W/B、AR/R）       │
│        ↓                                                │
│  ③ ref_model       参考镜像（含地址译码语义）            │
│        ↓                                                │
│  ④ scoreboard      期望 vs 实际，逐项 FAIL/PASS          │
│        ↓                                                │
│  ⑤ stats/report    统计与结果汇总                        │
└─────────────────────────────────────────────────────────┘
```

## 关键 API

| API | 作用 |
|---|---|
| `setup(limit, verbose, word_ma)` | 配置越界阈值、日志、地址译码模式 |
| `push_back(txn_t)` | 压入单笔事务 |
| `run_sequence(seq)` | 压入整个序列对象 |
| `execute_all()` | 顺序执行全部事务 |
| `report()` | 打印统计，返回 PASS/FAIL |
| `axi_sequence.add/add_loopback/add_random` | 构造激励 |

## `txn_t` 字段 → VIP 行为

| 字段 | VIP 行为 |
|---|---|
| `write` | 1=走 AW/W/B，0=走 AR/R |
| `addr` | 按 `word_ma` 决定译码（默认 `addr>>2` 字寻址） |
| `data` | 写数据 / 读回比对基准 |
| `pause` | 事务后插入 N 拍总线空闲 |
| `same` | 写事务 AW 与 W 是否同拍 |
| `exp_err` | 期望 SLVERR（越界向量） |

## 实测结果（Verilator 5.055）

```
== VIP demo: AXI->APB->AXI (NREG=16) ==
[VIP] txns=67 (W=31 R=36) checks=67 errors=0 pause=91 SLVERR=2
VIP PASS: 67 checks, 0 errors
```

- 67 笔事务（31 写 / 36 读），含 91 拍 pause，2 笔越界 SLVERR
- 覆盖：写后读回、背靠背、同拍写 + pause、60 笔随机、越界读写

## 调试记录：VIP 发现真实模型偏差

首次运行报 **16 个读数据不匹配**，根因：

- **DUT 从机**：`idx = PADDR >> 2`（字地址译码，地址 0/1/2/3 映射同一寄存器）
- **参考模型**：以字节地址为键（0/1/2/3 视为 4 个独立位置）

两者语义不一致 → 修复为参考模型也按 `addr >> 2` 寻址，并可通过 `word_ma` 参数切换。这正是 VIP 的价值：**验证模型与 DUT 语义必须对齐，否则会误报**。

## 复现

```bash
cd verilog/axi

# 编译检查（slang）
slang --timescale=1ns/1ps --top tb_vip_demo \
  -I sv -I vip -I . \
  axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
  sv/axi_lite_if.sv sv/apb_if.sv \
  sv/axi_apb_bridge_sv.sv sv/apb_slave_regs_sv.sv \
  vip/axi_vip_pkg.sv vip/axi_lite_vip.sv vip/tb_vip_demo.sv

# 仿真（Verilator ≥5.055）
verilator --binary --assert --timing --top-module tb_vip_demo \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-ASCRANGE -Wno-MULTIDRIVEN \
  -I sv -I vip -I . \
  axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
  sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
  sv/apb_slave_regs_sv.sv vip/axi_vip_pkg.sv vip/axi_lite_vip.sv \
  vip/tb_vip_demo.sv -o run/tb_vip_demo && ./run/tb_vip_demo
```

## UVM 版本（服务器）

```bash
xrun -sv -uvm -uvmhome $UVM_HOME \
  axi_vip_uvm_pkg.sv tb_uvm_top.sv
# 或
vcs -sverilog -ntb_opts uvm-1.2 axi_vip_uvm_pkg.sv tb_uvm_top.sv
```

UVM 组件对应：`axi_txn`→`uvm_sequence_item`、`axi_base_seq`→`uvm_sequence`、
`axi_driver`→`uvm_driver`、`axi_monitor`→`uvm_monitor`、
`axi_scoreboard`→`uvm_scoreboard`、`axi_vip_env`→`uvm_env`。

## 接入新 DUT 的检查清单

1. DUT 的 AXI4-Lite slave 端口接 `axi_lite_if`（slave modport）
2. 实例化 `axi_lite_vip`，端口接同一 interface
3. `setup(.limit(地址空间上限))` —— **越界阈值必须与 DUT 地址空间一致**
4. `setup(.word_ma(1))` —— **地址译码模式必须与 DUT 一致**（字/字节寻址）
5. 构造 sequence：先 loopback 验证基本读写，再加随机与边界
6. `report()` 检查；`exp_err=1` 的向量用于验证错误响应