# verilog：DUT + test bench

## 目录

| 文件 | 说明 |
|---|---|
| `counter.v` | 4-bit 同步计数器（DUT）：同步复位 / 使能 / 回绕 |
| `tb_counter.v` | counter 的 test bench（20 项断言，iverilog 可跑） |
| `counter8.sv` | 8-bit 同步计数器（新版 DUT）：增加 carry 进位输出 |
| `tb_counter8_icarus.sv` | counter8 test bench，**iverilog 12 可运行**（26 项断言） |
| `tb_counter8_sv.sv` | counter8 完整 SV test bench（interface/SVA/covergroup/class 随机），**需 irun/VCS** |
| `run_tb.sh` / `run_tb8.sh` | 一键仿真脚本 |
| `plot_waveform.py` | VCD → PNG 波形绘图（matplotlib） |

## 运行

```bash
bash run_tb.sh    # counter  (4-bit, 20 项断言)
bash run_tb8.sh   # counter8 (8-bit, 26 项断言)
python3 plot_waveform.py run/tb_counter.vcd   # 波形图
```

## iverilog 12 的 SystemVerilog 支持边界（实测）

| 特性 | iverilog 12 | 完整 SV（irun/VCS） |
|---|---|---|
| `logic` / `always_ff` / `enum` / `typedef` | ✅ | ✅ |
| `interface` + `clocking` | ❌ clocking | ✅ |
| SVA 并发断言 `assert property` | ❌ | ✅ |
| `covergroup` / 覆盖率 | ❌ | ✅ |
| `class` + `rand` + `constraint` | ❌ | ✅ |

因此：
- 本机快速验证用 `tb_counter8_icarus.sv`（断言改为传统 task）
- 需要 SVA / 覆盖率 / 随机激励（UVM 前站）时必须上 **irun / VCS**，用 `tb_counter8_sv.sv`

## 激励的起始 / 结束形式（TB 生命周期）

每个 test bench 按 **startup → 阶段测试 → drain → finalize** 组织激励：

```
┌─ 起始激励 startup ─────────────────────────────────────────────
│  上电复位序列（power-on reset）：
│    rst_n=0 保持 2 周期 → 检查 count=0, carry=0   (DUT 全部归零)
│    rst_n=1 再观察 2 周期 → 确认稳定后再进入测试
│  作用：所有触发器从确定状态开始，避免 X 传播 / 初值不确定
├─ 阶段测试（确定性 + 随机，见各 TB）
├─ 结束激励 1 drain ─────────────────────────────────────────────
│  激励回到安全态（rst 释放 / en 关闭）→ 跑 3 周期冲洗边沿
│  末拍快照 count 再等 1 沿，断言 en=0 下保持（收尾保持检查）
├─ 结束激励 2 finalize ──────────────────────────────────────────
│  打印 SUMMARY（错误数 / 随机周期数）
│  0 错误 → $finish    （退出码 0）
│  有错误 → $fatal(1)  （退出码 1，CI / run 脚本可捕获）
└────────────────────────────────────────────────────────────────
```

**退出码语义**（iverilog 实测）：
- `$finish(n)` 不管参数多少，vvp 退出码恒为 **0**
- `$fatal(1)` → vvp 退出码 **1**（CI 里 `bash run_tb8.sh || echo FAIL` 能捕获）

**watchdog 防挂死**（iverilog 版，可选编译）：
```bash
iverilog -g2012 -DTB_WATCHDOG counter8.sv tb_counter8_icarus.sv
```
超时（500 us，远超正常仿真时长）未结束则 `$fatal(1)` 终止。