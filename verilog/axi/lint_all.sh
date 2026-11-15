#!/usr/bin/env bash
# ============================================================================
# lint_all.sh —— 统一静态检查入口
# ============================================================================
# 本项目已多次踩坑的几类问题，做成静态检查：
#   握手时序  —— 只看 VALID 不看 READY 同拍（踩坑 5 次）
#   时钟竞争  —— 沿处立即阻塞驱动接口信号（踩坑 4 次）
#
# 用法: bash lint_all.sh
# 任一检查失败则 exit 1
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1

FAIL=0

if [ -t 1 ]; then R='\033[0m'; G='\033[32m'; Y='\033[31m'; B='\033[1m'; D='\033[2m'
else R=''; G=''; Y=''; B=''; D=''; fi

echo -e "${B}静态检查${R}"

run_check() {
    local name="$1" script="$2"
    printf "  %-24s " "$name"
    local out
    out=$(python3 "$script" 2>&1)
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo -e "${G}✓ PASS${R}"
    else
        echo -e "${Y}✗ FAIL${R}"
        echo "$out" | grep -E "^  ✗|合计" | head -8 | sed 's/^/      /'
        FAIL=$((FAIL+1))
    fi
}

run_check "握手时序" lint_handshake.py
run_check "时钟竞争" lint_clocking.py

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo -e "  ${G}${B}全部静态检查通过${R}"
    exit 0
else
    echo -e "  ${Y}${B}$FAIL 项静态检查失败${R}"
    exit 1
fi
