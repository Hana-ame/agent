#!/usr/bin/env bash
# ============================================================================
# sim.sh —— 统一仿真入口（只用 Verilator，不再依赖 iverilog）
# ============================================================================
# 用法:
#   bash sim.sh                  # 跑全部套件
#   bash sim.sh counter          # 4-bit 计数器
#   bash sim.sh counter8         # 8-bit 计数器
#   bash sim.sh axi              # AXI→APB→AXI 行为级
#   bash sim.sh sv               # 完整 SV 事务层（234 向量）
#   bash sim.sh gaps             # 覆盖缺口套件（285 检查）
#   bash sim.sh dual             # 并发桥（34 检查）
#   bash sim.sh vip              # VIP（67 检查）
#   bash sim.sh wave             # 完整 SV + VCD 波形
#
# 环境:
#   VERILATOR_ROOT   Verilator 安装根目录（若 verilator 不在 PATH）
#   VERILATOR        指定 verilator 可执行文件路径
#   WAIT=0|16        慢速外设 wait states（gaps 套件用）
#   SEED=123         随机种子（gaps 套件用）
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

RUN=run
mkdir -p "$RUN"

# ---- 定位 Verilator ----
# 优先级：显式指定 > 项目自编译 5.055+ > 系统 PATH
if [ -n "${VERILATOR:-}" ]; then
    VLT="$VERILATOR"
elif [ -f "${VERILATOR_ROOT:-/tmp/verilator}/bin/verilator_bin" ]; then
    VLT="${VERILATOR_ROOT:-/tmp/verilator}/bin/verilator_bin"
    export VERILATOR_ROOT="${VERILATOR_ROOT:-/tmp/verilator}"
elif command -v verilator >/dev/null 2>&1; then
    VLT="$(command -v verilator)"
else
    echo "ERROR: 找不到 verilator。请安装，或设置 VERILATOR_ROOT / VERILATOR 环境变量。" >&2
    exit 1
fi
echo "== verilator: $VLT =="
"$VLT" --version 2>/dev/null | head -1

# 通用 lint 抑制（均为无害的位宽/范围提示）
NOWARN="-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-ASCRANGE -Wno-MULTIDRIVEN -Wno-PINCONNECTEMPTY -Wno-TIMESCALEMOD -Wno-DECLFILENAME -Wno-VARHIDDEN -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-CONSTRAINTIGN -Wno-fatal"

build_run() {
    local name="$1"; shift
    local top="$1"; shift
    local extra="$1"; shift || true
    # shellcheck disable=SC2086
    echo "== sim: $name (top=$top) =="
    local out_abs
    out_abs="$(pwd)/$RUN/$name"
    # shellcheck disable=SC2086
    "$VLT" --binary --assert --timing --top-module "$top" \
        --Mdir "$RUN/obj_$name" $NOWARN $extra \
        "$@" -o "$out_abs" 2>&1 | grep -E '^%Error' && {
            echo "COMPILE FAILED: $name" >&2; return 1; }
    "$out_abs"
}

case "${1:-all}" in
    counter)
        build_run counter tb_counter "" counter.v tb_counter.v
        ;;
    counter8)
        build_run counter8 tb_counter8 "" counter8.sv tb_counter8_icarus.sv
        ;;
    axi)
        ( cd axi && build_run axi tb_axi_apb "" \
            axi_defs_pkg.sv apb_defs_pkg.sv apb_slave_regs.sv \
            axi_apb_bridge.sv axi_lite_vip_master.sv tb_axi_apb.sv )
        ;;
    sv)
        ( cd axi && build_run sv tb_axi_apb_sv "-I./sv -I." \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv sv/axi_txn_host.sv sv/tb_axi_apb_sv.sv )
        ;;
    gaps)
        ( cd axi && build_run gaps tb_gaps_sv \
            "-I./sv -I. -GWAIT_CYCLES=${WAIT:-2} -GSEED=${SEED:-0}" \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv sv/axi_txn_host.sv sv/tb_gaps_sv.sv )
        ;;
    dual)
        ( cd axi && build_run dual tb_dual_sv "-I./sv -I." \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_dual_sv.sv \
            sv/apb_slave_regs_sv.sv sv/tb_dual_sv.sv )
        ;;
    vip)
        ( cd axi && build_run vip tb_vip_demo "-I./sv -I./vip -I." \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv vip/axi_vip_pkg.sv vip/axi_lite_vip.sv \
            vip/tb_vip_demo.sv )
        ;;
    wave)
        ( cd axi && build_run wave tb_axi_apb_sv "-I./sv -I. --trace" \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv sv/axi_txn_host.sv sv/tb_axi_apb_sv.sv
          echo "VCD: $(ls axi/*.vcd 2>/dev/null || echo '(检查 axi 目录)')" )
        ;;
    all)
        echo "################ 全量回归 ################"
        bash "$0" counter
        bash "$0" counter8
        bash "$0" axi
        bash "$0" sv
        bash "$0" gaps
        bash "$0" dual
        bash "$0" vip
        echo "################ 全部完成 ################"
        ;;
    *)
        echo "未知目标: $1"; echo "可用: counter counter8 axi sv gaps dual vip wave all"
        exit 1
        ;;
esac
