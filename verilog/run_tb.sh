#!/usr/bin/env bash
# 编译并运行 counter test bench
# 产物统一落在 verilog/run/ 下（gitignore），不在仓库根目录乱放
set -euo pipefail
cd "$(dirname "$0")"

RUN=run
mkdir -p "$RUN"

iverilog -g2012 -o "$RUN/tb_counter.vvp" counter.v tb_counter.v
vvp "$RUN/tb_counter.vvp"