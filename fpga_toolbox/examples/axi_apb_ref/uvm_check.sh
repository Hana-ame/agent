#!/usr/bin/env bash
# ============================================================================
# uvm_check.sh —— 真实 UVM 库编译检查
# ============================================================================
# 背景：
#   README 声称「slang + 真实 UVM 库编译通过（0 errors）」，
#   但这个结论此前**没有任何自动化验证**，也没记录必需的命令参数。
#   实测发现：必须加 --timescale=1ns/1ps，否则 UVM 库自带 timescale
#   而项目文件没有，slang 会报 5-6 处 "design element does not have
#   a time scale defined"。
#
# 本脚本：
#   1. 自动定位 UVM 库源码（多个候选路径）
#   2. 用 slang 编译 UVM 包 + 项目 UVM VIP + 顶层 TB
#   3. 输出明确结论；找不到 UVM 库时降级为 SKIP（不算失败）
#
# 用法:
#   bash uvm_check.sh
#   UVM_SRC=/path/to/uvm-1.1d bash uvm_check.sh   # 显式指定
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1

if [ -t 1 ]; then R='\033[0m'; G='\033[32m'; Y='\033[31m'; B='\033[1m'; D='\033[2m'
else R=''; G=''; Y=''; B=''; D=''; fi

echo -e "${B}真实 UVM 库编译检查${R}"

# ---- 1. 定位 UVM 库 ----
find_uvm() {
    # 用于测试 SKIP 路径
    [ "${NO_UVM:-0}" -eq 1 ] && return 1
    if [ -n "${UVM_SRC:-}" ] && [ -f "$UVM_SRC/src/uvm_pkg.sv" ]; then
        echo "$UVM_SRC"; return 0
    fi
    # 常见候选：Questa/ModelSim 安装目录下的 verilog_src
    local base
    for base in /run/wsl/virtiofs-mounts/*/*/modeltech*/verilog_src \
                /opt/*/verilog_src /usr/local/*/verilog_src; do
        for v in uvm-1.1d uvm-1.1c uvm-1.1b uvm-1.2; do
            if [ -f "$base/$v/src/uvm_pkg.sv" ]; then
                echo "$base/$v"; return 0
            fi
        done
    done
    return 1
}

UVM=$(find_uvm)
if [ -z "$UVM" ]; then
    echo -e "  ${D}- SKIP${R} 未找到 UVM 库源码（README 的 UVM 编译结论无法复验）"
    echo -e "  ${D}  提示：设置 UVM_SRC=/path/to/uvm-1.1d 可显式指定${R}"
    exit 0
fi
echo -e "  UVM 库: ${D}$UVM${R}"

SLANG=$(command -v slang 2>/dev/null || true)
if [ -z "$SLANG" ]; then
    echo -e "  ${Y}✗${R} 未找到 slang，无法编译检查"
    exit 1
fi

# ---- 2. 编译 ----
# 关键：必须传 --timescale。UVM 库文件自带 `timescale，
# 而项目文件没有 → slang 报 "design element does not have a time scale"。
cd axi || exit 1
LOG=/tmp/uvm_check.log
if timeout 300 "$SLANG" --timescale=1ns/1ps \
       -I "$UVM/src" -I vip --top tb_uvm_top \
       "$UVM/src/uvm_pkg.sv" \
       axi_defs_pkg.sv apb_defs_pkg.sv \
       sv/bus_txn_pkg.sv sv/axi_lite_if.sv sv/apb_if.sv \
       sv/axi_apb_bridge_sv.sv sv/apb_slave_regs_sv.sv \
       vip/axi_vip_uvm_pkg.sv vip/tb_uvm_top.sv >"$LOG" 2>&1; then
    local_err=$(grep -cE "^.*error:" "$LOG" || true)
    echo -e "  ${G}✓${R} 真实 UVM 库编译通过（0 errors）"
    echo -e "  ${D}  uvm_pkg.sv + axi_vip_uvm_pkg.sv + tb_uvm_top.sv${R}"
    exit 0
else
    echo -e "  ${Y}✗${R} UVM 编译失败"
    grep -E "error:" "$LOG" | head -6 | sed 's/^/      /'
    echo -e "  ${D}  详见 $LOG${R}"
    exit 1
fi
