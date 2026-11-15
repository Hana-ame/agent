#!/usr/bin/env bash
# ============================================================================
# audit.sh —— 验证工具自审计：确认测试工具链真的能发现失败
# ============================================================================
# 背景：
#   上轮发现变异框架会「静默跳过」未还原的变异，导致得分虚高。
#   这提示一个更普遍的风险：如果一个验证工具在失败时静默通过，
#   它比没有工具更糟糕。
#
#   本脚本对每条验证链路做「故障注入 → 确认会报错 → 恢复」的闭环审计：
#     1. sim.sh 单套件   —— 注入缺陷后必须 exit != 0 且输出 FAIL
#     2. sim.sh all      —— 注入缺陷后必须中断且 exit != 0
#     3. ci.sh G3        —— 必须报告失败而不是全绿
#     4. mutate.py       —— 残留变异必须被检出（exit 4）
#     5. monitor         —— 协议监视器必须能抓到违规
#
#  每次注入后都用 trap 保证恢复，避免污染工作区。
#
# 用法:
#   bash audit.sh           # 跑全部审计
#   bash audit.sh --list    # 列出审计项
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1

SV=axi/sv
BACKUP_DIR=$(mktemp -d /tmp/audit_bak.XXXXXX)
PASS=0
FAIL=0
declare -a RESULTS=()

if [ -t 1 ]; then
    R='\033[0m'; GRN='\033[32m'; RED='\033[31m'; BOLD='\033[1m'; DIM='\033[2m'
else
    R=''; GRN=''; RED=''; BOLD=''; DIM=''
fi

banner() {
    echo ""
    echo -e "${BOLD}────────────────────────────────────────────────────────${R}"
    echo -e "${BOLD} $*${R}"
    echo -e "${BOLD}────────────────────────────────────────────────────────${R}"
}

record() {  # record <name> <PASS|FAIL> <detail>
    if [ "$2" = "PASS" ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
    RESULTS+=("$2|$1|$3")
}

# ---- 备份与恢复（保证工作区不被污染）----
# 用一个清单文件记录「备份文件 → 原始路径」的映射，
# 避免 cleanup 时无法知道该恢复到哪（曾因把 ci.sh 误恢复到 sv/ 而污染目录）。
MANIFEST="$BACKUP_DIR/manifest.txt"
: > "$MANIFEST"
BACKUP_FILE=""
CUR_ORIG=""

backup() {
    CUR_ORIG="$1"
    BACKUP_FILE="$BACKUP_DIR/$(echo "$1" | tr '/' '_')"
    cp "$1" "$BACKUP_FILE"
    echo "$BACKUP_FILE|$1" >> "$MANIFEST"
}
restore() {
    if [ -n "$BACKUP_FILE" ] && [ -f "$BACKUP_FILE" ] && [ -n "$CUR_ORIG" ]; then
        cp "$BACKUP_FILE" "$CUR_ORIG"
    fi
    BACKUP_FILE=""
    CUR_ORIG=""
}
cleanup() {
    # 兜底：按清单恢复到原始路径
    if [ -f "$MANIFEST" ]; then
        while IFS='|' read -r b org; do
            [ -f "$b" ] && [ -n "$org" ] && cp "$b" "$org"
        done < "$MANIFEST"
    fi
    rm -rf "$BACKUP_DIR"
}
trap cleanup EXIT

# ===========================================================================
# A1：单套件必须能报告失败
# ===========================================================================
audit_a1() {
    banner "A1 单套件失败检测（注入越界检查缺陷）"
    local f="$SV/apb_slave_regs_sv.sv"
    backup "$f"
    # 注入：越界检查恒真（非法地址也被接受）
    sed -i "s/assign in_range = (idx < NREG);/assign in_range = 1'b1;/" "$f"
    local out rc
    out=$(bash sim.sh gaps 2>&1); rc=$?
    restore "$f"
    if [ $rc -ne 0 ] && echo "$out" | grep -q "^FAIL"; then
        echo -e "  ${GRN}✓${R} gaps 套件正确报告失败（exit=$rc）"
        record "A1 单套件失败检测" PASS "exit=$rc"
    else
        echo -e "  ${RED}✗${R} gaps 套件未报告失败（exit=$rc）"
        record "A1 单套件失败检测" FAIL "exit=$rc"
    fi
}

# ===========================================================================
# A2：all 目标必须中断且返回非零
# ===========================================================================
audit_a2() {
    banner "A2 全量回归中断检测（注入同样的缺陷）"
    local f="$SV/apb_slave_regs_sv.sv"
    backup "$f"
    sed -i "s/assign in_range = (idx < NREG);/assign in_range = 1'b1;/" "$f"
    local out rc passes
    out=$(bash sim.sh all 2>&1); rc=$?
    restore "$f"
    passes=$(echo "$out" | grep -cE "^PASS|^VIP PASS")
    if [ $rc -ne 0 ] && [ "$passes" -lt 17 ]; then
        echo -e "  ${GRN}✓${R} all 正确中断（exit=$rc，仅 $passes/17 通过）"
        record "A2 全量回归中断检测" PASS "exit=$rc, ${passes}/17"
    else
        echo -e "  ${RED}✗${R} all 未正确中断（exit=$rc，$passes/17 通过）"
        record "A2 全量回归中断检测" FAIL "exit=$rc, ${passes}/17"
    fi
}

# ===========================================================================
# A3：CI G3 必须报告失败
# ===========================================================================
audit_a3() {
    banner "A3 CI G3 失败上报（注入缺陷）"
    local f="$SV/apb_slave_regs_sv.sv"
    backup "$f"
    sed -i "s/assign in_range = (idx < NREG);/assign in_range = 1'b1;/" "$f"
    local out rc
    out=$(bash ci.sh --stage 3 2>&1); rc=$?
    restore "$f"
    if [ $rc -ne 0 ] && echo "$out" | grep -q "FAIL"; then
        echo -e "  ${GRN}✓${R} G3 正确上报失败（exit=$rc）"
        record "A3 CI G3 失败上报" PASS "exit=$rc"
    else
        echo -e "  ${RED}✗${R} G3 未上报失败（exit=$rc）"
        record "A3 CI G3 失败上报" FAIL "exit=$rc"
    fi
}

# ===========================================================================
# A4：变异框架必须检出残留变异
# ===========================================================================
audit_a4() {
    banner "A4 变异框架残留检测（人为留下未还原变异）"
    local f="$SV/apb_slave_regfile_sv.sv"
    backup "$f"
    # 制造 M19 未还原的状态
    sed -i "s|return (attr == A_RSVD) ? '0 : cur_v;|return cur_v;|" "$f"
    local rc
    ( cd axi && python3 mutate.py --list >/dev/null 2>&1 ); rc=$?
    restore "$f"
    if [ "$rc" -eq 4 ]; then
        echo -e "  ${GRN}✓${R} 残留变异被检出（exit=4）"
        record "A4 变异框架残留检测" PASS "exit=4"
    else
        echo -e "  ${RED}✗${R} 残留变异未被检出（exit=$rc，期望 4）"
        record "A4 变异框架残留检测" FAIL "exit=$rc"
    fi
}

# ===========================================================================
# A5：协议监视器必须能抓到违规
# ===========================================================================
audit_a5() {
    banner "A5 协议监视器违规检测（自测套件）"
    local out rc
    out=$(bash sim.sh monitor 2>&1); rc=$?
    if [ $rc -eq 0 ] && echo "$out" | grep -q "PASS: monitor detects violations"; then
        echo -e "  ${GRN}✓${R} 监视器自测通过（能抓违规 + 不误报）"
        record "A5 协议监视器违规检测" PASS "自测通过"
    else
        echo -e "  ${RED}✗${R} 监视器自测失败（exit=$rc）"
        record "A5 协议监视器违规检测" FAIL "exit=$rc"
    fi
}

# ===========================================================================
# A6：CI 覆盖率阈值必须能拒绝低覆盖率
# ===========================================================================
audit_a6() {
    banner "A6 CI 覆盖率阈值拒绝（临时把阈值设为不可能达到）"
    local f=ci.sh
    backup "$f"
    sed -i 's/^MIN_RTL_COVERAGE=.*/MIN_RTL_COVERAGE=99.9/' "$f"
    local rc
    ( bash ci.sh --stage 5 >/dev/null 2>&1 ); rc=$?
    restore "$f"
    if [ $rc -ne 0 ]; then
        echo -e "  ${GRN}✓${R} 覆盖率 92.2% 被 99.9% 阈值正确拒绝（exit=$rc）"
        record "A6 覆盖率阈值拒绝" PASS "exit=$rc"
    else
        echo -e "  ${RED}✗${R} 阈值未生效（exit=$rc）"
        record "A6 覆盖率阈值拒绝" FAIL "exit=$rc"
    fi
}

# ===========================================================================
list_items() {
    echo "审计项："
    echo "  A1  单套件失败检测"
    echo "  A2  全量回归中断检测"
    echo "  A3  CI G3 失败上报"
    echo "  A4  变异框架残留检测"
    echo "  A5  协议监视器违规检测"
    echo "  A6  CI 覆盖率阈值拒绝"
}

main() {
    case "${1:-}" in
        --list) list_items; exit 0 ;;
        --help|-h) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        "") ;;
        *) echo "未知参数: $1（用 --list）" >&2; exit 2 ;;
    esac

    echo -e "${BOLD}验证工具自审计${R}  $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${DIM}目的：确认每条验证链路在失败时都会报错，而不是静默通过${R}"

    audit_a1
    audit_a2
    audit_a3
    audit_a4
    audit_a5
    audit_a6

    banner "审计结果"
    for r in "${RESULTS[@]}"; do
        local st="${r%%|*}"; local rest="${r#*|}"; local nm="${rest%%|*}"; local dt="${rest#*|}"
        if [ "$st" = "PASS" ]; then
            printf "  ${GRN}✓ PASS${R}  %-28s ${DIM}%s${R}\n" "$nm" "$dt"
        else
            printf "  ${RED}✗ FAIL${R}  %-28s ${DIM}%s${R}\n" "$nm" "$dt"
        fi
    done

    echo ""
    if [ "$FAIL" -eq 0 ]; then
        echo -e "  ${GRN}${BOLD}全部 $PASS 项审计通过${R} —— 验证工具在失败时都会报错"
        echo ""
        exit 0
    else
        echo -e "  ${RED}${BOLD}$FAIL 项审计失败${R} —— 存在静默通过风险"
        echo ""
        exit 1
    fi
}

main "$@"
