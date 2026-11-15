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
#   bash audit.sh --list         # 列出审计项
#   bash audit.sh --allow-dirty  # 跳过「工作区必须干净」的前置检查
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1

SV=axi/sv
BACKUP_DIR=$(mktemp -d /tmp/audit_bak.XXXXXX)
PASS=0
FAIL=0
checks_total=0
ALLOW_DIRTY=0
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
        # 立即校验恢复是否成功（曾因恢复失败导致 RTL 残留缺陷，
        # 进而让后续全量回归失败 —— 与变异框架曾踩的坑同类）
        if ! cmp -s "$BACKUP_FILE" "$CUR_ORIG"; then
            echo "  ❌ 严重：$CUR_ORIG 恢复失败，立即中止" >&2
            exit 3
        fi
    fi
    BACKUP_FILE=""
    CUR_ORIG=""
}
cleanup() {
    # 兜底：按清单恢复到原始路径，并校验恢复成功
    local restore_fail=0
    if [ -f "$MANIFEST" ]; then
        while IFS='|' read -r b org; do
            [ -f "$b" ] && [ -n "$org" ] && cp "$b" "$org"
            if [ -f "$b" ] && [ -n "$org" ] && ! cmp -s "$b" "$org"; then
                restore_fail=1
                echo "  ❌ 恢复失败：$org" >&2
            fi
        done < "$MANIFEST"
    fi
    if [ "$restore_fail" -eq 1 ]; then
        echo "  ⚠ 部分文件恢复失败，请用 git status / git diff 检查" >&2
        echo "  提示：git checkout -- verilog/axi/sv/" >&2
    fi
    rm -rf "$BACKUP_DIR"
}
# 监听多个信号：仅 EXIT 不足以覆盖被中断的情况。
# 注意：若中断发生在 sleep/长时间外部命令期间，bash 的 trap 行为取决于
# 信号是否传递给 bash 本身。这里显式监听 INT/TERM/HUP 提高覆盖率，
# 并在 cleanup 里做幂等处理（多次调用安全）。
TRAP_ACTIVE=1
cleanup_once() {
    # 幂等：只执行一次
    if [ "${TRAP_ACTIVE:-0}" -eq 1 ]; then
        TRAP_ACTIVE=0
        cleanup
    fi
}
trap cleanup_once EXIT INT TERM HUP

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
    local out rc passes total
    # 动态统计套件总数，避免硬编码数字随项目增长而过时
    total=$(sed -n '/^    all)/,/^        ;;/p' sim.sh | grep -cE 'bash "\$0"')
    [ "${total:-0}" -eq 0 ] && total=1
    out=$(bash sim.sh all 2>&1); rc=$?
    restore "$f"
    passes=$(echo "$out" | grep -cE "^PASS|^VIP PASS")
    if [ $rc -ne 0 ] && [ "$passes" -lt "$total" ]; then
        echo -e "  ${GRN}✓${R} all 正确中断（exit=$rc，仅 $passes/$total 通过）"
        record "A2 全量回归中断检测" PASS "exit=$rc, ${passes}/$total"
    else
        echo -e "  ${RED}✗${R} all 未正确中断（exit=$rc，$passes/$total 通过）"
        record "A2 全量回归中断检测" FAIL "exit=$rc, ${passes}/$total"
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
    # 阈值必须「物理上不可能达到」。覆盖率已达 100%，
    # 故用 101（此前用 99.9 已无法触发拒绝，属于审计用例失效）
    sed -i 's/^MIN_RTL_COVERAGE=.*/MIN_RTL_COVERAGE=101/' "$f"
    local rc
    ( bash ci.sh --stage 5 >/dev/null 2>&1 ); rc=$?
    restore "$f"
    if [ $rc -ne 0 ]; then
        echo -e "  ${GRN}✓${R} 覆盖率被 101% 阈值正确拒绝（exit=$rc）"
        record "A6 覆盖率阈值拒绝" PASS "exit=$rc"
    else
        echo -e "  ${RED}✗${R} 阈值未生效（exit=$rc）"
        record "A6 覆盖率阈值拒绝" FAIL "exit=$rc"
    fi
}

# ===========================================================================
# A7：仿真挂死时输出必须保留（stdbuf 生效）
# ===========================================================================
audit_a7() {
    banner "A7 挂死输出保留（诊断可靠性）"
    local work=/tmp/audit_a7
    rm -rf "$work"; mkdir -p "$work"

    # 造一个会挂死的迷你 TB
    cat > "$work/t_hang.sv" <<'VEOF'
`timescale 1ns/1ps
module t_hang;
  logic clk=0;
  always #5 clk=~clk;
  initial begin
    $display("HANG-MARK-1");
    repeat(3) @(posedge clk);
    $display("HANG-MARK-2");
    forever @(posedge clk);
  end
endmodule
VEOF

    local vlt="${VERILATOR:-/tmp/verilator/bin/verilator_bin}"
    export VERILATOR_ROOT=${VERILATOR_ROOT:-/tmp/verilator}
    if ! "$vlt" --binary --timing --top-module t_hang \
         --Mdir "$work/obj" -Wno-fatal "$work/t_hang.sv" \
         -o "$work/tb" >/dev/null 2>&1; then
        echo -e "  ${RED}✗${R} 无法构建测试用例"
        record "A7 挂死输出保留" FAIL "build"
        return 1
    fi

    # 无缓冲：应丢失输出
    local n_nobuf n_buf
    n_nobuf=$(timeout 3 "$work/tb" 2>&1 | grep -c "HANG-MARK" || true)
    # 有缓冲：应保留输出
    n_buf=$(timeout 3 stdbuf -o0 -e0 "$work/tb" 2>&1 | grep -c "HANG-MARK" || true)

    if [ "${n_buf:-0}" -ge 2 ]; then
        echo -e "  ${GRN}✓${R} stdbuf 保留输出（无缓冲时 ${n_nobuf:-0} 行，有缓冲 ${n_buf} 行）"
        record "A7 挂死输出保留" PASS "buf=$n_buf nobuf=${n_nobuf:-0}"
    else
        echo -e "  ${RED}✗${R} stdbuf 未生效（仅 ${n_buf:-0} 行）"
        record "A7 挂死输出保留" FAIL "buf=${n_buf:-0}"
    fi
    # 验证 sim.sh 已使用 stdbuf
    checks_total=$((checks_total+1))
    if grep -q "stdbuf -o0" sim.sh; then
        echo -e "  ${GRN}✓${R} sim.sh 仿真运行已启用 stdbuf"
    else
        echo -e "  ${RED}✗${R} sim.sh 未启用 stdbuf"
        record "A7 sim.sh stdbuf" FAIL "missing"
    fi
    rm -rf "$work"
}

# ===========================================================================
# A8：CI 门槛「能拒绝」的集中验证
# ===========================================================================
# 背景：A6 曾因覆盖率升到 100% 而使 99.9% 阈值无法触发，
# 审计长期显示 PASS 却什么也没验证。这类「门槛失效」很隐蔽。
# A8 集中验证所有数值型门槛在「不可能达到的阈值」下都会拒绝。
# ===========================================================================
audit_a8() {
    banner "A8 CI 门槛有效性（阈值不可达时必须拒绝）"
    local f=ci.sh
    local ok=0 bad=0

    check_gate() {
        local name="$1" var="$2" orig="$3" bad_val="$4"
        local rc
        sed -i "s/^${var}=.*/${var}=${bad_val}/" "$f"
        if grep -q "^${var}=${bad_val}" "$f"; then
            ( bash ci.sh --stage "$5" >/dev/null 2>&1 ); rc=$?
            sed -i "s/^${var}=.*/${var}=${orig}/" "$f"
            if [ $rc -ne 0 ]; then
                echo -e "  ${GRN}✓${R} $name：阈值 ${bad_val} 被正确拒绝（exit=$rc）"
                ok=$((ok+1))
            else
                echo -e "  ${RED}✗${R} $name：阈值 ${bad_val} 未生效（exit=$rc）"
                bad=$((bad+1))
            fi
        else
            echo -e "  ${RED}✗${R} $name：无法注入阈值（变量 $var 不存在？）"
            bad=$((bad+1))
        fi
    }

    backup "$f"
    # G5 覆盖率：阈值 101（覆盖率最高 100）
    check_gate "G5 覆盖率阈值" MIN_RTL_COVERAGE 90 101 5
    # G6 变异得分：阈值 100（本项目得分 90.9）
    check_gate "G6 变异得分阈值" MIN_MUTATION_SCORE 88 100 6
    restore "$f"

    # 注入后确认 ci.sh 已完全恢复
    checks_total=$((checks_total+1))
    if grep -q '^MIN_RTL_COVERAGE=90' ci.sh && grep -q '^MIN_MUTATION_SCORE=88' ci.sh; then
        echo -e "  ${GRN}✓${R} ci.sh 阈值已恢复原值"
    else
        echo -e "  ${RED}✗${R} ci.sh 阈值未恢复，请检查"
        bad=$((bad+1))
    fi

    if [ "$bad" -eq 0 ]; then
        record "A8 CI 门槛有效性" PASS "$ok 个门槛全部有效"
    else
        record "A8 CI 门槛有效性" FAIL "$bad 个门槛失效"
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
    echo "  A7  挂死输出保留（stdbuf）"
    echo "  A8  CI 门槛有效性（集中验证）"
}

main() {
    case "${1:-}" in
        --allow-dirty) ALLOW_DIRTY=1; shift; main "$@" ;;
        --list) list_items; exit 0 ;;
        --help|-h) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        "") ;;
        *) echo "未知参数: $1（用 --list）" >&2; exit 2 ;;
    esac

    # 前置检查：工作区不应有未提交的 RTL 改动（防止上一轮注入残留）
    # --allow-dirty 用于「开发中/被其他工具调用」场景（如 scripts_check.sh），
    # 此时工作区本来就可能不干净，不应因此判定审计失败。
    local dirty
    if [ "${ALLOW_DIRTY:-0}" -eq 1 ]; then
        echo -e "  ${DIM}（--allow-dirty：跳过工作区前置检查）${R}"
        dirty=""
    else
        dirty=$(git -C .. status --porcelain -- verilog/axi/sv/ 2>/dev/null | grep -v "^ M verilog/axi/sv/tb_" || true)
    fi
    if [ -n "$dirty" ]; then
        echo -e "${RED}❌ 前置检查失败：RTL 目录存在未提交改动${R}" >&2
        echo "$dirty" | sed 's/^/   /' >&2
        echo "   可能原因：上一轮审计/变异注入未完全恢复。请先用 git diff 检查。" >&2
        exit 4
    fi

    echo -e "${BOLD}验证工具自审计${R}  $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${DIM}目的：确认每条验证链路在失败时都会报错，而不是静默通过${R}"

    audit_a1
    audit_a2
    audit_a3
    audit_a4
    audit_a5
    audit_a6
    audit_a7
    audit_a8

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
