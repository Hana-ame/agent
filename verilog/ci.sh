#!/usr/bin/env bash
# ============================================================================
# ci.sh —— 质量门：按阶段验证项目质量，任一关失败即停止
# ============================================================================
# 设计原则：
#   1. 每关独立、可单独运行（--stage N）
#   2. 失败立即停止并打印诊断（不掩盖问题）
#   3. 输出机器可读摘要（便于 CI 集成）
#   4. 总耗时可控（默认跳过最慢的变异测试，需显式开启）
#
# 关卡设计（从快到慢，快速失败）：
#   G1 编译检查   slang 检查全部 SV（语法/语义，最快，秒级）
#   G2 快速回归   4 个基础套件（counter/counter8/axi/sv）
#   G3 全量回归   11 个套件全部
#   G4 参数扫掠   18 组种子×wait 组合
#   G5 覆盖率     RTL 有效覆盖率达标（阈值 90%）
#   G6 变异测试   变异得分达标（阈值 80%）
#   G7 文档生成   6 个 HTML 全部生成成功
#
# 用法:
#   bash ci.sh                  # 跑 G1-G3 + G7（默认，约 3 分钟）
#   bash ci.sh --full           # 跑全部关卡（含 sweep/coverage/mutate，约 25 分钟）
#   bash ci.sh --stage 5        # 只跑第 5 关
#   bash ci.sh --list           # 列出所有关卡
# ============================================================================
set -o pipefail
cd "$(dirname "$0")" || exit 1

# ---- 配置 ----
MIN_RTL_COVERAGE=90      # G5 阈值（%）
MIN_MUTATION_SCORE=88    # G6 阈值（%）
SLANG=$(command -v slang 2>/dev/null || echo "")
VERILATOR="${VERILATOR:-/tmp/verilator/bin/verilator_bin}"
[ -z "$VERILATOR_ROOT" ] && export VERILATOR_ROOT=/tmp/verilator

# ---- 输出 ----
if [ -t 1 ]; then
    R='\033[0m'; BOLD='\033[1m'; GRN='\033[32m'; RED='\033[31m'
    YEL='\033[33m'; CYN='\033[36m'; DIM='\033[2m'
else
    R=''; BOLD=''; GRN=''; RED=''; YEL=''; CYN=''; DIM=''
fi

STAGE_RESULTS=()
STAGE_TIMES=()

banner() {
    echo ""
    echo -e "${BOLD}${CYN}════════════════════════════════════════════════════════${R}"
    echo -e "${BOLD}${CYN}  $*${R}"
    echo -e "${BOLD}${CYN}════════════════════════════════════════════════════════${R}"
}

record() {   # record <name> <PASS|FAIL|SKIP> <seconds>
    STAGE_RESULTS+=("$2|$1")
    STAGE_TIMES+=("$3")
}

# 关卡定义
STAGE_NAMES=(
    "G1:编译检查(slang)"
    "G2:快速回归(4套件)"
    "G3:全量回归(20套件)"
    "G4:参数扫掠(18组合)"
    "G5:RTL覆盖率(>=${MIN_RTL_COVERAGE}%)"
    "G6:变异得分(>=${MIN_MUTATION_SCORE}%)"
    "G7:文档生成(6份HTML)"
)

list_stages() {
    echo "可用关卡："
    local i=1
    for n in "${STAGE_NAMES[@]}"; do
        printf "  %d  %s\n" "$i" "$n"
        i=$((i+1))
    done
    echo ""
    echo "默认执行: G1 G2 G3 G7"
    echo "--full 执行: 全部 7 关"
}

# ---------------------------------------------------------------------------
# G1：slang 编译检查（全部 SV 文件）
# ---------------------------------------------------------------------------
stage1() {
    banner "G1 编译检查（slang）"
    if [ -z "$SLANG" ]; then
        echo -e "${YEL}SKIP: 未找到 slang${R}"
        record "$1" SKIP 0
        return 0
    fi
    local t0=$SECONDS
    local cnt_file=/tmp/ci_g1_count
    : > "$cnt_file"

    # 检查主回归 + 各专项的编译
    # 注意：此函数在子 shell 中被调用，用文件记录结果（变量无法跨 shell）
    check() {
        local label="$1"; shift
        if "$SLANG" --timescale=1ns/1ps "$@" >/tmp/ci_slang.log 2>&1; then
            echo -e "  ${GRN}✓${R} $label"
            echo "PASS" >> "$cnt_file"
        else
            echo -e "  ${RED}✗${R} $label"
            tail -6 /tmp/ci_slang.log | sed 's/^/      /'
            echo "FAIL" >> "$cnt_file"
        fi
    }

    ( cd axi || exit 1
      check "主回归 tb_axi_apb_sv" --top tb_axi_apb_sv -I sv -I . \
        axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
        sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
        sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv \
        sv/axi_txn_host.sv sv/tb_axi_apb_sv.sv
      check "gaps 套件" --top tb_gaps_sv -I sv -I . \
        axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
        sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
        sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv \
        sv/axi_txn_host.sv sv/tb_gaps_sv.sv
      check "并发桥 dual" --top tb_dual_sv -I sv -I . \
        axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
        sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_dual_sv.sv \
        sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv sv/tb_dual_sv.sv
      check "协议监视器自测" --top tb_mon_selftest -I sv -I . \
        axi_defs_pkg.sv apb_defs_pkg.sv sv/axi_lite_if.sv \
        sv/apb_if.sv sv/axi_apb_monitor.sv sv/tb_monitor_selftest.sv
      check "监视器规则逐条" --top tb_monitor_rules -I sv -I . \
        axi_defs_pkg.sv apb_defs_pkg.sv sv/axi_lite_if.sv \
        sv/apb_if.sv sv/axi_apb_monitor.sv sv/tb_monitor_rules.sv
      check "延迟基线" --top tb_latency_sv -I sv -I . \
        axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
        sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
        sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv sv/tb_latency_sv.sv
      check "仲裁公平性" --top tb_fairness_sv -I sv -I . \
        axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
        sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_dual_sv.sv \
        sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv sv/tb_fairness_sv.sv
      check "参数矩阵" --top tb_param_sv -I sv -I . -DTXN_DW=32 \
        axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
        sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
        sv/axi_apb_bridge_dual_sv.sv \
        sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv \
        sv/axi_txn_host.sv sv/tb_param_sv.sv
      check "轻量UVM环境" --top tb_uvm_lite_top -I sv -I . -I vip \
        axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
        sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
        sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv \
        vip/axi_uvm_lite_pkg.sv vip/tb_uvm_lite_top.sv
      check "SVA 前件激活" --top tb_sva_antecedent_sv -I sv -I . \
        axi_defs_pkg.sv apb_defs_pkg.sv sv/axi_lite_if.sv \
        sv/apb_if.sv sv/tb_sva_antecedent_sv.sv
      check "断言有效性(APB)" --top tb_assert_effectiveness_sv -I sv -I . -DCASE=5 \
        axi_defs_pkg.sv apb_defs_pkg.sv sv/apb_if.sv \
        sv/tb_assert_effectiveness_sv.sv
      check "断言有效性(AXI)" --top tb_axi_assert_eff_sv -I sv -I . -DACASE=0 \
        axi_defs_pkg.sv apb_defs_pkg.sv sv/axi_lite_if.sv \
        sv/apb_if.sv sv/tb_axi_assert_eff_sv.sv
      check "DW=64 数据通路" --top tb_dw64_sv -I sv -I . \
        axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
        sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
        sv/apb_slave_regs_sv.sv sv/tb_dw64_sv.sv
      check "寄存器生成物" --top tb_regfile_gen -I sv -I . \
        axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
        sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
        sv/apb_slave_regfile_gen.sv sv/axi_apb_monitor.sv \
        sv/axi_txn_host.sv sv/tb_regfile_gen.sv
    )

    local dt=$((SECONDS-t0))
    local total fails
    total=$(wc -l < "$cnt_file" 2>/dev/null | tr -d ' ')
    fails=$(grep -c FAIL "$cnt_file" 2>/dev/null)
    fails=${fails:-0}
    fails=$(echo "$fails" | head -1 | tr -d ' \n')
    if [ "${total:-0}" -eq 0 ]; then
        echo -e "  ${RED}${BOLD}FAIL${R} —— 没有执行任何编译检查（${dt}s）"
        record "$1" FAIL $dt
        return 1
    fi
    if [ "$fails" -eq 0 ]; then
        echo -e "  ${GRN}${BOLD}PASS${R} —— $total/$total 项编译通过（${dt}s）"
        record "$1" PASS $dt
        return 0
    else
        echo -e "  ${RED}${BOLD}FAIL${R} —— $fails/$total 项失败（${dt}s）"
        record "$1" FAIL $dt
        return 1
    fi
}

# ---------------------------------------------------------------------------
# G2：快速回归
# ---------------------------------------------------------------------------
stage2() {
    banner "G2 快速回归（4 个基础套件）"
    local t0=$SECONDS
    local fails=0
    for s in counter counter8 axi sv; do
        printf "  %-10s " "$s"
        out=$(bash sim.sh "$s" 2>&1)
        rc=$?
        # 必须同时满足：退出码 0 + 有明确的 PASS 行（防止空跑/缓存导致的假通过）
        if [ $rc -eq 0 ] && echo "$out" | grep -qE "^PASS|^VIP PASS"; then
            # 提取检查数作为证据
            local ev
            ev=$(echo "$out" | grep -oE "[0-9]+ (checks|checks passed|txns)" | head -1)
            echo -e "${GRN}✓${R} ${DIM}${ev}${R}"
        else
            echo -e "${RED}✗${R} (exit=$rc)"
            echo "$out" | tail -5 | sed 's/^/      /'
            fails=$((fails+1))
        fi
    done
    local dt=$((SECONDS-t0))
    if [ $fails -eq 0 ]; then
        echo -e "  ${GRN}${BOLD}PASS${R} —— 4/4 套件通过（${dt}s）"
        record "$1" PASS $dt
        return 0
    else
        echo -e "  ${RED}${BOLD}FAIL${R} —— $fails/4 套件失败（${dt}s）"
        record "$1" FAIL $dt
        return 1
    fi
}

# ---------------------------------------------------------------------------
# G3：全量回归
# ---------------------------------------------------------------------------
stage3() {
    banner "G3 全量回归（11 个套件）"
    local t0=$SECONDS
    local out
    out=$(bash sim.sh all 2>&1)
    local rc=$?
    local passes
    passes=$(echo "$out" | grep -cE "^PASS|^VIP PASS")
    echo "$out" | grep -E "^== sim|^PASS|^VIP PASS|^FAIL" | sed 's/^/  /'
    # 汇总检查数作为「确实执行了测试」的证据
    local total_checks
    total_checks=$(echo "$out" | grep -oE "[0-9]+ (checks|checks passed)" \
                   | grep -oE "^[0-9]+" | awk '{s+=$1} END{print s+0}')
    local dt=$((SECONDS-t0))
    if [ $rc -eq 0 ] && [ "$passes" -ge 20 ] && [ "${total_checks:-0}" -ge 1000 ]; then
        echo -e "  ${GRN}${BOLD}PASS${R} —— $passes/20 套件通过，累计 ${total_checks} 项检查（${dt}s）"
        record "$1" PASS $dt
        return 0
    else
        echo -e "  ${RED}${BOLD}FAIL${R} —— $passes/20 通过，检查数 ${total_checks:-0}（${dt}s）"
        record "$1" FAIL $dt
        return 1
    fi
}

# ---------------------------------------------------------------------------
# G4：参数扫掠
# ---------------------------------------------------------------------------
stage4() {
    banner "G4 参数扫掠（9 种子 × 9 wait = 18 组合）"
    local t0=$SECONDS
    local out
    out=$(bash sim.sh sweep 2>&1)
    local rc=$?
    echo "$out" | grep -E "^  (SEED|WAIT)|^PASS|^FAIL" | sed 's/^/  /'
    local dt=$((SECONDS-t0))
    if [ $rc -eq 0 ]; then
        echo -e "  ${GRN}${BOLD}PASS${R} —— 18/18 组合通过（${dt}s）"
        record "$1" PASS $dt
        return 0
    else
        echo -e "  ${RED}${BOLD}FAIL${R} —— 参数扫掠有失败（${dt}s）"
        record "$1" FAIL $dt
        return 1
    fi
}

# ---------------------------------------------------------------------------
# G5：RTL 覆盖率阈值
# ---------------------------------------------------------------------------
stage5() {
    banner "G5 RTL 覆盖率（阈值 ${MIN_RTL_COVERAGE}%）"
    local t0=$SECONDS
    local out
    out=$(bash sim.sh coverage 2>&1)
    local rc=$?
    # 提取「RTL 有效覆盖率: NN.N%」或 Python 摘要的同类行
    local pct
    pct=$(echo "$out" | grep -oE "RTL.*有效覆盖率[:：][[:space:]]*[0-9.]+" | grep -oE "[0-9.]+$" | tail -1)
    if [ -z "$pct" ]; then
        # 回退：从 gen_coverage_html.py 的摘要行提取
        pct=$(echo "$out" | grep -oE "有效覆盖率:[[:space:]]*[0-9.]+" | grep -oE "[0-9.]+$" | tail -1)
    fi
    echo "$out" | grep -E "RTL|真实零覆盖|注解缺陷" | sed 's/^/  /'
    local dt=$((SECONDS-t0))
    if [ $rc -ne 0 ] || [ -z "$pct" ]; then
        echo -e "  ${RED}${BOLD}FAIL${R} —— 覆盖率收集失败或无法解析（${dt}s）"
        record "$1" FAIL $dt
        return 1
    fi
    local ok
    ok=$(awk -v p="$pct" -v m="$MIN_RTL_COVERAGE" 'BEGIN{print (p>=m)?1:0}')
    if [ "$ok" -eq 1 ]; then
        echo -e "  ${GRN}${BOLD}PASS${R} —— RTL 有效覆盖率 ${pct}% ≥ ${MIN_RTL_COVERAGE}%（${dt}s）"
        record "$1" PASS $dt
        return 0
    else
        echo -e "  ${RED}${BOLD}FAIL${R} —— RTL 有效覆盖率 ${pct}% < ${MIN_RTL_COVERAGE}%（${dt}s）"
        record "$1" FAIL $dt
        return 1
    fi
}

# ---------------------------------------------------------------------------
# G6：变异得分阈值
# ---------------------------------------------------------------------------
stage6() {
    banner "G6 变异得分（阈值 ${MIN_MUTATION_SCORE}%）"
    local t0=$SECONDS
    local out
    out=$(cd axi && python3 mutate.py 2>&1)
    local rc=$?
    echo "$out" | grep -E "^  [✅❌⚠]|变异得分|存活" | sed 's/^/  /'
    local pct
    pct=$(echo "$out" | grep -oE "变异得分:[[:space:]]*[0-9.]+" | grep -oE "[0-9.]+$" | tail -1)
    local dt=$((SECONDS-t0))
    if [ -z "$pct" ]; then
        echo -e "  ${RED}${BOLD}FAIL${R} —— 无法解析变异得分（${dt}s）"
        record "$1" FAIL $dt
        return 1
    fi
    local ok
    ok=$(awk -v p="$pct" -v m="$MIN_MUTATION_SCORE" 'BEGIN{print (p>=m)?1:0}')
    if [ "$ok" -eq 1 ]; then
        echo -e "  ${GRN}${BOLD}PASS${R} —— 变异得分 ${pct}% ≥ ${MIN_MUTATION_SCORE}%（${dt}s）"
        record "$1" PASS $dt
        return 0
    else
        echo -e "  ${RED}${BOLD}FAIL${R} —— 变异得分 ${pct}% < ${MIN_MUTATION_SCORE}%（${dt}s）"
        record "$1" FAIL $dt
        return 1
    fi
}

# ---------------------------------------------------------------------------
# G7：文档生成
# ---------------------------------------------------------------------------
stage7() {
    banner "G7 文档生成（6 份 HTML）"
    local t0=$SECONDS
    local fails=0
    ( cd axi || exit 1
      for gen in gen_fsm_html.py gen_interface_map.py gen_test_plan.py gen_regfile.py gen_coverage_html.py gen_portal.py; do
          printf "  %-26s " "$gen"
          if python3 "$gen" >/tmp/ci_gen.log 2>&1; then
              echo -e "${GRN}✓${R}"
          else
              echo -e "${RED}✗${R}"
              tail -4 /tmp/ci_gen.log | sed 's/^/      /'
              fails=$((fails+1))
          fi
      done
    )
    # 校验 6 份文件存在且非空
    local expect=(DESIGN.html INTERFACE_MAP.html TEST_PLAN.html COVERAGE.html FSM.html INDEX.html)
    local missing=0
    for f in "${expect[@]}"; do
        if [ ! -s "axi/$f" ]; then
            echo -e "  ${RED}✗${R} 缺少 $f"
            missing=$((missing+1))
        fi
    done
    local dt=$((SECONDS-t0))
    if [ $fails -eq 0 ] && [ $missing -eq 0 ]; then
        echo -e "  ${GRN}${BOLD}PASS${R} —— 6/6 份文档生成（${dt}s）"
        record "$1" PASS $dt
        return 0
    else
        echo -e "  ${RED}${BOLD}FAIL${R} —— 生成失败 $fails，缺失 $missing（${dt}s）"
        record "$1" FAIL $dt
        return 1
    fi
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
main() {
    local run_stages=(1 2 3 7)
    local i

    case "${1:-}" in
        --list) shift; list_stages; exit 0 ;;
        --full) run_stages=(1 2 3 4 5 6 7) ;;
        --stage)
            local s="${2:-}"
            if ! [[ "$s" =~ ^[1-7]$ ]]; then
                echo "错误：--stage 需要 1-7" >&2; exit 2
            fi
            run_stages=("$s")
            ;;
        --help|-h) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        "") ;;
        *) echo "未知参数: $1（用 --help）" >&2; exit 2 ;;
    esac

    echo -e "${BOLD}质量门 CI${R}  $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${DIM}Verilator: $VERILATOR${R}"
    echo -e "${DIM}slang:     ${SLANG:-未找到}${R}"
    echo -e "${DIM}关卡:      ${run_stages[*]}${R}"

    local start=$SECONDS
    local failed_at=""
    for i in "${run_stages[@]}"; do
        local name="${STAGE_NAMES[$((i-1))]}"
        if ! "stage$i" "$name"; then
            failed_at="$name"
            break     # 快速失败
        fi
    done
    local total=$((SECONDS-start))

    # ---- 汇总 ----
    banner "质量门结果"
    local k=0
    for r in "${STAGE_RESULTS[@]}"; do
        local st="${r%%|*}" nm="${r#*|}"
        local tm="${STAGE_TIMES[$k]}"
        case "$st" in
            PASS) printf "  ${GRN}✓ PASS${R}  %-32s %3ss\n" "$nm" "$tm" ;;
            SKIP) printf "  ${YEL}- SKIP${R}  %-32s %3ss\n" "$nm" "$tm" ;;
            *)    printf "  ${RED}✗ FAIL${R}  %-32s %3ss\n" "$nm" "$tm" ;;
        esac
        k=$((k+1))
    done

    echo ""
    if [ -n "$failed_at" ]; then
        echo -e "  ${RED}${BOLD}质量门未通过${R} —— 卡在 $failed_at（总耗时 ${total}s）"
        echo ""
        exit 1
    else
        echo -e "  ${GRN}${BOLD}质量门全部通过${R}（总耗时 ${total}s）"
        echo ""
        exit 0
    fi
}

main "$@"
