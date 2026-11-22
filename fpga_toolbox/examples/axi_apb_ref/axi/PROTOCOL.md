# AXI → APB → AXI 协议定义与转换设计

本目录实现一条完整链路：**AXI4-Lite master → AXI-APB 桥（转换设计）→ 慢速 APB 外设 → 响应回 AXI**。
核心目标："可以慢，但一定要合法"——慢速外设可以插入任意 wait states，但总线时序必须完全符合协议。

## 目录

| 文件 | 角色 |
|---|---|
| `axi_defs_pkg.sv`  | **完整 AXI 定义**（package） |
| `apb_defs_pkg.sv`  | **完整 APB 定义**（package） |
| `axi_apb_bridge.sv` | **转换设计**：AXI4-Lite → APB 单事务串行桥 |
| `apb_slave_regs.sv` | APB 慢速从机（寄存器堆 + wait states + 越界 SLVERR） |
| `axi_lite_vip_master.sv` | AXI master 激励（TB 用） |
| `tb_axi_apb.sv`     | **testbench**：数据一致性 + 协议合法性 monitor |

## 1. 完整 AXI 定义（axi_defs_pkg）

五通道 struct 全部字段：

```
写地址 AW: { awid, awaddr, awlen, awsize, awburst, awlock, awcache, awprot, awqos, awregion }
写数据 W : { wdata, wstrb, wlast }
写响应 B : { bid, bresp }
读地址 AR: { arid, araddr, arlen, arsize, arburst, arlock, arcache, arprot, arqos, arregion }
读数据 R : { rid, rdata, rresp, rlast }
全局     : ACLK, ARESETn
```

协议常量：

| 项 | 值 |
|---|---|
| `RESP_OKAY` / `EXOKAY` / `SLVERR` / `DECERR` | 2'b00 / 01 / 10 / 11 |
| `BURST_FIXED` / `INCR` / `WRAP` | 2'b00 / 01 / 10 |
| `SIZE_BYTE` / `HALF` / `WORD` / `DWRD` | 0 / 1 / 2 / 3 |
| `LOCK_NORMAL` / `LOCK_EXCL` | 0 / 1 |

协议函数（DUT / 参考模型共用单一事实来源）：

- `resp_name(r)` — 响应编码 → 助记名
- `is_aligned(addr, size)` — 按 AxSIZE 检查地址对齐
- `valid_channels_ok(awv, wv, arv)` — 单事务桥的通道请求合法性

## 2. 完整 APB 定义（apb_defs_pkg）

总线信号（打包 struct `apb_bus_t`）：`psel, penable, pwrite, paddr, pwdata, pready, prdata, pslverr`（另 PCLK/PRESETn 全局）。

阶段状态机（`apb_phase_t`）：`IDLE → SETUP → ACC → IDLE`

| 阶段 | 条件 | 行为 |
|---|---|---|
| IDLE | PSEL=0, PENABLE=0 | 无传输 |
| SETUP | PSEL=1, PENABLE=0 | 地址/方向有效，至少 1 拍 |
| ACC | PSEL=1, PENABLE=1 | 数据相位，等 PREADY |

**时序规则（"合法"的形式化）**，均可调用 `apb_defs_pkg` 函数检查：

| 规则 | 内容 | 函数 |
|---|---|---|
| R1 | PENABLE 只能在 PSEL=1 时拉高 | `penable_ok` |
| R2 | ACCESS 未完成（PREADY=0）不得中途退出 | `phase_transition_ok` |
| R3 | SETUP 期间从机不拉高 PREADY | （monitor 检查） |
| R4 | 传输在 `PSEL&&PENABLE&&PREADY` 完成 | `transfer_done` |
| R5 | 完成后 PENABLE/PSEL 归 0，下一传输重新 SETUP | （从机实现） |
| R6 | 写数据完成拍锁存；读数据完成拍后有效 | （从机实现） |
| R7 | PSLVERR 只在完成拍有效 | （从机实现） |

## 3. 转换设计（axi_apb_bridge）

单状态机串行化，一个事务只走一条路径：

```
写：IDLE → AW/W 握手 → W_SETUP → W_ACCESS(等 PREADY) → W_RESP(BVALID) → IDLE
读：IDLE → AR 握手    → R_SETUP → R_ACCESS(等 PREADY) → R_RESP(RVALID)  → IDLE
```

- 支持 AW/W 同拍（`AXI4-Lite` 允许）或分开发（`AW_HELD`/`W_HELD` 等待）
- APB 慢速从机插 wait states 时，桥在 `W_ACCESS/R_ACCESS` 完整等待（"可以慢"）
- APB `PSLVERR`（越界）→ 桥转成 `BRESP/RRESP = RESP_SLVERR`（2'b10）
- 响应只在完成拍后给出，无提前响应、无重复响应

## 4. testbench（tb_axi_apb）

- **数据一致性**：16 寄存器随机写 → 读回逐字比对（参考模型镜像 `ref_mem`）
- **慢速合法**：`WAIT_CYCLES=3`，monitor 全程检查 APB 规则 R1/R2/R3
- **越界响应**：`0x40`（reg16）读/写 → 期望 `SLVERR`，且用 `resp_name` 打印确认
- **同拍写**：AW+W 同拍正确

运行：

```bash
bash ../sim.sh axi
# => PASS: axi->apb->axi all 35 checks passed (monitor clean)
```

## 5. 工具兼容性注记

本目录所有文件用 Verilator 5.055 可编译执行（统一工具链）：


完整 SV（interface 端口 + clocking 驱动 + UVM 风格）在实际 EDA 工具（irun/VCS）上运行时可再扩展。