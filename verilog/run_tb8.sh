#!/usr/bin/env bash
# counter8 新版 test bench：本机 iverilog 可运行版
set -euo pipefail
cd "$(dirname "$0")"

RUN=run
mkdir -p "$RUN"

iverilog -g2012 -o "$RUN/tb_counter8.vvp" counter8.sv tb_counter8_icarus.sv
vvp "$RUN/tb_counter8.vvp"