#!/usr/bin/env bash
# 服务器运行脚本：在装有 EDA 工具的机器上跑完整 SV test bench
# 用法:
#   bash run_sv.sh            # 自动探测 irun/xrun/vcs
#   TOOL=irun bash run_sv.sh  # 强制指定
#   TOOL=vcs  bash run_sv.sh
set -euo pipefail
cd "$(dirname "$0")"

RUN=run
mkdir -p "$RUN"
SRCS="counter8.sv tb_counter8_sv.sv"
TOOL="${TOOL:-auto}"

pick_tool() {
  if command -v xrun >/dev/null; then echo xrun; return; fi
  if command -v irun  >/dev/null; then echo irun;  return; fi
  if command -v vcs   >/dev/null; then echo vcs;   return; fi
  echo "none"
}

TOOL="$( [ "$TOOL" = auto ] && pick_tool || echo "$TOOL" )"
echo "== tool: $TOOL =="

case "$TOOL" in
  xrun|irun)
    # Cadence Xcelium
    xrun -sv \
      -timescale 1ns/1ps \
      -coverage all \
      -covoverwrite \
      -XMLibdir "$RUN/xcelium.d" \
      "$SRCS" \
      -access +rwc \
      -gui -input run.tcl || true  # 失败也继续结束（方便看错误）
    ;;
  vcs)
    vcs -sverilog \
      -timescale=1ns/1ps \
      -cm line+cond+branch+assert \
      -cm_dir "$RUN/simv.vdb" \
      "$SRCS" \
      -o "$RUN/simv" \
      && "$RUN/simv" -cm line+cond+branch+assert
    ;;
  none)
    echo "ERROR: no EDA tool found (need xrun/irun or vcs)"
    exit 1
    ;;
  *)
    echo "ERROR: unknown TOOL=$TOOL"
    exit 1
    ;;
esac