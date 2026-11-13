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