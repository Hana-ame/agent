#!/usr/bin/env bash
# 完整 SV 版（verilog/axi/sv/）的 EDA 运行脚本
# 用法:
#   bash run_sv_chain.sh            # 自动探测 irun/xrun/vcs
#   TOOL=irun bash run_sv_chain.sh  # 强制 irun (Cadence Xcelium)
#   TOOL=vcs  bash run_sv_chain.sh  # 强制 vcs (Synopsys)
#   TOOL=slang bash run_sv_chain.sh # 仅编译检查（本机可用）
#   TOOL=iverilog bash run_sv_chain.sh # 仅编译定义包（iverilog 子集）
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=axi/sv/run
mkdir -p "$RUN"
TOOL="${TOOL:-auto}"

# 编译单元：定义包 + 完整 SV 链路
# 注意：脚本 cd 到 verilog/ 目录，故前缀 axi/
VERILOG_SRCS="axi/axi_defs_pkg.sv axi/apb_defs_pkg.sv \
  axi/sv/bus_txn_pkg.sv \
  axi/sv/axi_lite_if.sv axi/sv/apb_if.sv \
  axi/sv/axi_apb_bridge_sv.sv axi/sv/apb_slave_regs_sv.sv \
  axi/sv/axi_txn_host.sv \
  axi/sv/tb_axi_apb_sv.sv"

pick_tool() {
  # 优先 slang（完整 SV 编译检查），其次 EDA 工具
  if command -v xrun >/dev/null; then echo xrun; return; fi
  if command -v irun  >/dev/null; then echo irun;  return; fi
  if command -v vcs   >/dev/null; then echo vcs;   return; fi
  if command -v slang  >/dev/null; then echo slang;  return; fi
  echo "none"
}
TOOL="$( [ "$TOOL" = auto ] && pick_tool || echo "$TOOL" )"
echo "== run_sv_chain.sh: tool=$TOOL =="

case "$TOOL" in
  xrun|irun)
    # Cadence Xcelium：SV + 断言 + 覆盖率
    xrun -sv \
      -timescale 1ns/1ps \
      -coverage all \
      -covoverwrite \
      -xmlibdirname "$RUN/xcelium.d" \
      -assert enable_diag \
      $VERILOG_SRCS \
      -l "$RUN/xrun.log"
    ;;
  vcs)
    # Synopsys VCS：SV + 断言 + 覆盖率
    vcs -sverilog \
      -timescale=1ns/1ps \
      -cm line+cond+branch+assert \
      -cm_dir "$RUN/simv.vdb" \
      -assert enable_diag \
      $VERILOG_SRCS \
      -o "$RUN/simv" \
      -l "$RUN/vcs.log" \
      && "$RUN/simv" -cm line+cond+branch+assert
    ;;
  slang)
    # 完整 SV 编译检查（本机可验证语法语义，等价 EDA 编译）
    slang --timescale=1ns/1ps --top tb_axi_apb_sv \
      -I axi/sv -I axi \
      $VERILOG_SRCS
    echo "== slang: BUILD OK (编译检查通过；仿真需 EDA 工具) =="
    ;;
  iverilog)
    # iverilog 只能编译定义包（完整 SV 链路不支持 interface 端口）
    iverilog -g2012 -o /dev/null axi/axi_defs_pkg.sv axi/apb_defs_pkg.sv
    echo "== iverilog: 定义包编译 OK（完整 SV 需 slang/EDA） =="
    ;;
  none|*)
    echo "ERROR: 未找到 EDA 工具。请安装 slang 做编译检查，或"
    echo "       在装有 irun/xrun/vcs 的服务器上运行本脚本。"
    echo "提示: 本仓库工具链—— slangs 见 README，EDA 见 192.168.1.53"
    exit 1
    ;;
esac