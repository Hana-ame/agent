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

# verilator_coverage 通常在 verilator 同目录
VLT_DIR="$(dirname "$VLT")"
if [ -x "$VLT_DIR/verilator_coverage" ]; then
    VLT_COV="$VLT_DIR/verilator_coverage"
elif command -v verilator_coverage >/dev/null 2>&1; then
    VLT_COV="$(command -v verilator_coverage)"
else
    VLT_COV=""
fi

# 通用 lint 抑制（均为无害的位宽/范围提示）
NOWARN="-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-ASCRANGE -Wno-MULTIDRIVEN -Wno-PINCONNECTEMPTY -Wno-TIMESCALEMOD -Wno-DECLFILENAME -Wno-VARHIDDEN -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-CONSTRAINTIGN -Wno-fatal"

build_run() {
    local name="$1"
    local top="$2"
    local extra="$3"
    shift 3
    local srcs=("$@")
    echo "== sim: $name (top=$top) =="
    # 在「当前工作目录」下创建 run（支持 axi 等子目录：
    # 顶层只创建了 verilog/run，子目录运行时会缺 axi/run，
    # 干净 clone 后首次运行会报 "Can't write file: run/obj_xxx/..."）
    mkdir -p "$RUN"
    local out_abs
    out_abs="$(pwd)/$RUN/$name"   # 绝对路径（Verilator 在 obj 子目录 make）
    # shellcheck disable=SC2086
    if ! "$VLT" --binary --assert --timing --top-module "$top" \
            --Mdir "$RUN/obj_$name" $NOWARN $extra \
            "${srcs[@]}" -o "$out_abs" 2>&1 | grep -E '^%Error'; then
        :   # 无 %Error 输出，视为成功
    else
        echo "COMPILE FAILED: $name" >&2
        return 1
    fi
    [ -x "$out_abs" ] || { echo "BUILD MISSING: $out_abs" >&2; return 1; }
    # 关键：禁用 stdout 缓冲再运行仿真。
    # 若仿真挂死被 kill，缓冲未 flush 会导致「零输出」，
    # 极易被误判为「程序启动即退出」（本项目已踩坑 4 次）。
    # stdbuf 不可用时回退到直接执行。
    if command -v stdbuf >/dev/null 2>&1; then
        stdbuf -o0 -e0 "$out_abs"
    else
        "$out_abs"
    fi
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
            sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv \
            sv/axi_txn_host.sv sv/tb_axi_apb_sv.sv )
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
            sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv \
            sv/tb_dual_sv.sv )
        ;;
    regfilegen)
        ( cd axi && build_run regfilegen tb_regfile_gen "-I./sv -I. -DTXN_DW=${DW:-32} -GDUAL=${DUAL:-0}" \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/axi_apb_bridge_dual_sv.sv \
            sv/apb_slave_regfile_gen.sv sv/axi_apb_monitor.sv \
            sv/axi_txn_host.sv sv/tb_regfile_gen.sv )
        ;;
    regfile)
        ( cd axi && build_run regfile tb_regfile_sv "-I./sv -I. -DTXN_DW=${DW:-32} -GDUAL=${DUAL:-0}" \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/axi_apb_bridge_dual_sv.sv \
            sv/apb_slave_regfile_sv.sv sv/axi_apb_monitor.sv \
            sv/axi_txn_host.sv sv/tb_regfile_sv.sv )
        ;;
    stress)
        ( cd axi && build_run stress tb_dual_stress "-I./sv -I." \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_dual_sv.sv \
            sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv sv/tb_dual_stress.sv )
        ;;
    monitor)
        ( cd axi && build_run monitor tb_mon_selftest "-I./sv -I." \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/axi_lite_if.sv \
            sv/apb_if.sv sv/axi_apb_monitor.sv sv/tb_monitor_selftest.sv )
        ;;
    params)
        # 参数配置矩阵：逐配置验证 DUT 在非默认参数下仍正确
        CNT=/tmp/params_cnt_sim
        : > "$CNT"
        run_cfg() {
            local aw="$1" dw="$2" nreg="$3" wc="$4" label="$5" dual="${6:-0}"
            local d="run/param_${aw}_${dw}_${nreg}_${wc}_d${dual}"
            rm -rf "$d"; mkdir -p "$d"
            if ! "$VLT" --binary --assert --timing --top-module tb_param_sv \
                 --Mdir "$d/obj" $NOWARN -I./sv -I. \
                 -GAW=$aw -GDW=$dw -GNREG=$nreg -GWAITC=$wc -DTXN_DW=$dw \
                 -GDUAL=$dual \
                 axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
                 sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
                 sv/axi_apb_bridge_dual_sv.sv \
                 sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv \
                 sv/axi_txn_host.sv sv/tb_param_sv.sv \
                 -o "$(pwd)/$d/tb" >/dev/null 2>&1; then
                printf "  %-14s AW=%-2s DW=%-2s NREG=%-2s WAIT=%-2s BUILD-FAIL\n" \
                       "$label" "$aw" "$dw" "$nreg" "$wc"
                echo "FAIL" >> "$CNT"; return
            fi
            local tb_out=""
            tb_out=$( cd "$d" && ./tb 2>&1 || true )
            if echo "$tb_out" | grep -q "^PASS"; then
                printf "  %-14s AW=%-2s DW=%-2s NREG=%-2s WAIT=%-2s PASS\n" \
                       "$label" "$aw" "$dw" "$nreg" "$wc"
                echo "PASS" >> "$CNT"
            else
                printf "  %-14s AW=%-2s DW=%-2s NREG=%-2s WAIT=%-2s FAIL\n" \
                       "$label" "$aw" "$dw" "$nreg" "$wc"
                echo "FAIL" >> "$CNT"
            fi
        }
        ( cd axi
          echo "== params: 参数配置矩阵 =="
          run_cfg 8 32 16 2 "串行默认"    0
          run_cfg 6 16 4  0 "串行小配置"  0
          run_cfg 4 8  2  0 "串行最小"    0
          run_cfg 10 64 16 4 "串行大位宽" 0
          run_cfg 8 32 32 8 "串行多寄存器" 0
          # 边界配置（本轮补充）：验证参数空间的极值
          run_cfg 1 32 1  0 "串行地址极窄" 0     # AW=1：最小合法地址位宽
          run_cfg 4 32 1  0 "串行单寄存器" 0     # NREG=1：地址译码边界
          run_cfg 4 8  2 15 "串行长反压"  0      # WAIT=15：最长等待
          run_cfg 8 32 16 2 "并发默认"    1
          run_cfg 6 16 4  0 "并发小配置"  1
          run_cfg 4 8  2  0 "并发最小"    1
          run_cfg 10 64 16 4 "并发大位宽" 1
          run_cfg 8 32 32 8 "并发多寄存器" 1
          run_cfg 1 32 1  0 "并发地址极窄" 1
          run_cfg 4 32 1  0 "并发单寄存器" 1
          run_cfg 4 8  2 15 "并发长反压"  1
        )
        echo "------------------------------------------------------"
        TOTAL=$(grep -c . "$CNT" 2>/dev/null || true); TOTAL=${TOTAL:-0}
        FAILED=$(grep -c FAIL "$CNT" 2>/dev/null || true); FAILED=${FAILED:-0}
        TOTAL=$(echo "$TOTAL" | head -1 | tr -d ' \n')
        FAILED=$(echo "$FAILED" | head -1 | tr -d ' \n')
        rm -f "$CNT"
        # 期望组数动态计算（硬编码会在新增配置时要求变松）
        EXPECT_CFG=$(sed -n '/^    params)/,/^        ;;/p' sim.sh \
                     | grep -cE '^\s+run_cfg [0-9]' || true)
        EXPECT_CFG=${EXPECT_CFG:-0}
        if [ "$FAILED" -eq 0 ] && [ "$TOTAL" -eq "$EXPECT_CFG" ] && [ "$TOTAL" -gt 0 ]; then
            echo "PASS: params —— $TOTAL 组参数配置全部通过（期望 $EXPECT_CFG）"
            exit 0
        else
            echo "FAIL: params —— $FAILED/$TOTAL 组失败（期望 $TOTAL=$EXPECT_CFG）"
            exit 1
        fi
        ;;
    asserteff)
        # 断言有效性：逐条注入违规，确认断言会失败（而非真空通过）
        ( cd axi
          set +e          # 本任务是逐条检查并汇总，不因单条失败而中断
          AE_FAIL=0
          echo "== asserteff: 断言有效性验证 =="
          for c in 1 2 3 4; do
              d="run/ae_$c"
              rm -rf "$d"; mkdir -p "$d"
              if ! "$VLT" --binary --assert --timing \
                   --top-module tb_assert_effectiveness_sv --Mdir "$d/obj" \
                   $NOWARN -I./sv -I. -DCASE=$c \
                   axi_defs_pkg.sv apb_defs_pkg.sv sv/apb_if.sv \
                   sv/tb_assert_effectiveness_sv.sv \
                   -o "$(pwd)/$d/tb" >/dev/null 2>&1; then
                  printf "  CASE=%-2s BUILD-FAIL\n" "$c"
                  AE_FAIL=$((AE_FAIL+1)); continue
              fi
              out=$( cd "$d" && ./tb 2>&1 || true )
              # 精确校验：必须是「预期的那条断言」触发，不能只看有没有报错。
              # 否则错误触发别的断言也会被判为 PASS。
              case $c in
                  1) AE_EXPECT="ap_penable_requires_psel" ;;
                  2) AE_EXPECT="ap_no_ready_in_setup" ;;
                  3) AE_EXPECT="ap_access_stable_until_ready" ;;
                  4) AE_EXPECT="ap_xfer_done_exits" ;;
              esac
              if echo "$out" | grep -qE "Assertion failed.*[.]${AE_EXPECT}:"; then
                  printf "  CASE=%-2s 断言 %-28s 正确触发 PASS\n" "$c" "$AE_EXPECT"
              elif echo "$out" | grep -q "Assertion failed"; then
                  got=$(echo "$out" | grep -oE "ap_[a-z_]+" | head -1)
                  printf "  CASE=%-2s 触发的是 %s，不是 %s FAIL\n" "$c" "$got" "$AE_EXPECT"
                  AE_FAIL=$((AE_FAIL+1))
              else
                  printf "  CASE=%-2s 断言 %-28s 未报错 FAIL\n" "$c" "$AE_EXPECT"
                  AE_FAIL=$((AE_FAIL+1))
              fi
          done
          # CASE 5：合法序列不应误报
          d="run/ae_5"; rm -rf "$d"; mkdir -p "$d"
          if "$VLT" --binary --assert --timing \
               --top-module tb_assert_effectiveness_sv --Mdir "$d/obj" \
               $NOWARN -I./sv -I. -DCASE=5 \
               axi_defs_pkg.sv apb_defs_pkg.sv sv/apb_if.sv \
               sv/tb_assert_effectiveness_sv.sv \
               -o "$(pwd)/$d/tb" >/dev/null 2>&1; then
              out=$( cd "$d" && ./tb 2>&1 || true )
              if echo "$out" | grep -q "^PASS"; then
                  printf "  CASE=5  合法序列无误报 PASS\n"
              else
                  printf "  CASE=5  合法序列误报 FAIL\n"
                  AE_FAIL=$((AE_FAIL+1))
              fi
          else
              printf "  CASE=5  BUILD-FAIL\n"
              AE_FAIL=$((AE_FAIL+1))
          fi
          # ---- AXI 5 条断言 ----
          for c in 1 2 3 4 5; do
              d="run/axae_$c"
              rm -rf "$d"; mkdir -p "$d"
              if ! "$VLT" --binary --assert --timing \
                   --top-module tb_axi_assert_eff_sv --Mdir "$d/obj" \
                   $NOWARN -I./sv -I. -DACASE=$c \
                   axi_defs_pkg.sv apb_defs_pkg.sv \
                   sv/axi_lite_if.sv sv/apb_if.sv \
                   sv/tb_axi_assert_eff_sv.sv \
                   -o "$(pwd)/$d/tb" >/dev/null 2>&1; then
                  printf "  AXICASE=%-2s BUILD-FAIL\n" "$c"
                  AE_FAIL=$((AE_FAIL+1)); continue
              fi
              out=$( cd "$d" && ./tb 2>&1 || true )
              case $c in
                  1) AE_EXPECT="ap_aw_valid_stable" ;;
                  2) AE_EXPECT="ap_w_valid_stable" ;;
                  3) AE_EXPECT="ap_ar_valid_stable" ;;
                  4) AE_EXPECT="ap_b_valid_stable" ;;
                  5) AE_EXPECT="ap_r_valid_stable" ;;
              esac
              if echo "$out" | grep -qE "Assertion failed.*[.]${AE_EXPECT}:"; then
                  printf "  AXICASE=%-2s 断言 %-28s 正确触发 PASS\n" "$c" "$AE_EXPECT"
              elif echo "$out" | grep -q "Assertion failed"; then
                  got=$(echo "$out" | grep -oE "ap_[a-z_]+" | head -1)
                  printf "  AXICASE=%-2s 触发的是 %s，不是 %s FAIL\n" "$c" "$got" "$AE_EXPECT"
                  AE_FAIL=$((AE_FAIL+1))
              else
                  printf "  AXICASE=%-2s 断言 %-28s 未报错 FAIL\n" "$c" "$AE_EXPECT"
                  AE_FAIL=$((AE_FAIL+1))
              fi
          done
          if [ "$AE_FAIL" -eq 0 ]; then
              echo "PASS: assertion effectiveness（APB 4 条 + AXI 5 条断言均能捕获违规，合法序列无误报）"
          else
              echo "FAIL: $AE_FAIL 项断言有效性检查失败"
              exit 1
          fi
        )
        ;;
    lint)
        # 静态检查：握手时序 + 时钟竞争
        ( cd axi && bash lint_all.sh )
        ;;
    stability)
        # 稳定性检查：每个套件重复运行，检测偶发失败
        python3 stability.py "${@:2}"
        ;;
    duallegal)
        ( cd axi && build_run duallegal tb_dual_legal_sv "-I./sv -I. -DTXN_DW=${DW:-32}" \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_dual_sv.sv \
            sv/apb_slave_regs_sv.sv sv/axi_txn_host.sv \
            sv/tb_dual_legal_sv.sv )
        ;;
    fieldiso)
        ( cd axi && build_run fieldiso tb_field_iso_sv "-I./sv -I. -DTXN_DW=${DW:-32}" \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regfile_gen.sv sv/axi_txn_host.sv \
            sv/tb_field_iso_sv.sv )
        ;;
    dualerr)
        ( cd axi && build_run dualerr tb_dual_err_sv "-I./sv -I. -DTXN_DW=${DW:-32}" \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_dual_sv.sv \
            sv/apb_slave_regs_sv.sv sv/axi_txn_host.sv \
            sv/tb_dual_err_sv.sv )
        ;;
    resetdom)
        ( cd axi && build_run resetdom tb_reset_domain_sv "-I./sv -I. -DTXN_DW=${DW:-32}" \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv sv/axi_txn_host.sv \
            sv/tb_reset_domain_sv.sv )
        ;;
    handshake)
        ( cd axi && build_run handshake tb_handshake_protocol_sv "-I./sv -I." \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv sv/tb_handshake_protocol_sv.sv )
        ;;
    uvmlite)
        ( cd axi && build_run uvmlite tb_uvm_lite_top "-I./sv -I. -I./vip" \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv \
            vip/axi_uvm_lite_pkg.sv vip/tb_uvm_lite_top.sv )
        ;;
    sva)
        ( cd axi && build_run sva tb_sva_antecedent_sv "-I./sv -I." \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/axi_lite_if.sv \
            sv/apb_if.sv sv/tb_sva_antecedent_sv.sv )
        ;;
    dw64)
        ( cd axi && build_run dw64 tb_dw64_sv "-I./sv -I." \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv sv/tb_dw64_sv.sv )
        ;;
    fairness)
        ( cd axi && build_run fairness tb_fairness_sv "-I./sv -I." \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_dual_sv.sv \
            sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv sv/tb_fairness_sv.sv )
        ;;
    latency)
        ( cd axi && build_run latency tb_latency_sv "-I./sv -I. -GWAITC=${WAIT:-2}" \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv sv/tb_latency_sv.sv )
        ;;
    monitorrules)
        ( cd axi && build_run monitorrules tb_monitor_rules "-I./sv -I." \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/axi_lite_if.sv \
            sv/apb_if.sv sv/axi_apb_monitor.sv sv/tb_monitor_rules.sv )
        ;;
    covgaps)
        ( cd axi && build_run covgaps tb_cov_gaps "--coverage -I./sv -I." \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv sv/tb_cov_gaps.sv )
        ;;
    vip)
        ( cd axi && build_run vip tb_vip_demo "-I./sv -I./vip -I." \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv vip/axi_vip_pkg.sv vip/axi_lite_vip.sv \
            vip/tb_vip_demo.sv )
        ;;
    coverage)
        # 覆盖率收集：编译两套件（--coverage），分别运行，合并报告
        COVDIR="run/cov"
        rm -rf "$COVDIR"; mkdir -p "$COVDIR"
        run_cov() {
            local top="$1"; shift
            local d="$COVDIR/$top"
            mkdir -p "$d"
            echo "== coverage: 编译 $top =="
            # shellcheck disable=SC2086
            "$VLT" --binary --assert --timing --coverage \
                --top-module "$top" --Mdir "$d/obj" $NOWARN \
                -I./sv -I. "$@" -o "$(pwd)/$d/tb" 2>&1 | grep -E '^%Error' && return 1
            echo "== coverage: 运行 $top =="
            ( cd "$d" && ./tb >/dev/null 2>&1; cp coverage.dat "cov_$top.dat" )
        }
        ( cd axi
          run_cov tb_axi_apb_sv \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv \
            sv/axi_txn_host.sv sv/tb_axi_apb_sv.sv
          run_cov tb_cov_gaps \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv sv/tb_cov_gaps.sv
          run_cov tb_dual_stress \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_dual_sv.sv \
            sv/apb_slave_regs_sv.sv sv/tb_dual_stress.sv
          run_cov tb_regfile_gen \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regfile_gen.sv sv/axi_apb_monitor.sv \
            sv/axi_txn_host.sv sv/tb_regfile_gen.sv
          # monitor 规则测试：唯一会触发 violation() 分支的套件。
          # 不纳入覆盖率收集的话，监视器的检查逻辑会显示为大面积零覆盖，
          # 那是「收集范围问题」而非「代码不可达」。
          run_cov tb_monitor_rules \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv \
            sv/tb_monitor_rules.sv
          cp -r "$COVDIR"/*/cov_*.dat "$COVDIR/" 2>/dev/null || true
        )
        echo "== coverage: 合并报告 =="
        # 注解需在源码所在目录生成（文件名相对路径），故 cd axi
        if [ -n "$VLT_COV" ]; then
            ( cd axi && "$VLT_COV" --annotate "run/cov/annotated" run/cov/cov_*.dat 2>&1 | head -12 )
            # 自动生成 HTML 覆盖率报告
            if [ -f axi/gen_coverage_html.py ]; then
                echo "== coverage: 生成 HTML 报告 =="
                ( cd axi && python3 gen_coverage_html.py 2>&1 | tail -9 )
            fi
        else
            echo "WARN: 未找到 verilator_coverage" >&2
        fi
        ;;
    sweep)
        # 参数扫掠：多种子 × 多 wait cycles，验证测试在参数空间稳定
        FAILED=0
        TOTAL=0
        echo "== sweep: 多种子（WAIT=${WAIT:-2}）=="
        for s in 0 1 7 42 123 999 31337 271828 161803; do
            TOTAL=$((TOTAL+1))
            if SEED=$s bash "$0" gaps 2>&1 | grep -q "^PASS"; then
                printf "  SEED=%-7s PASS\n" "$s"
            else
                printf "  SEED=%-7s FAIL\n" "$s"
                FAILED=$((FAILED+1))
            fi
        done
        echo "== sweep: 多 wait cycles（SEED=${SEED:-0}）=="
        for w in 0 1 2 3 4 8 16 31 63; do
            TOTAL=$((TOTAL+1))
            if WAIT=$w bash "$0" gaps 2>&1 | grep -q "^PASS"; then
                printf "  WAIT=%-7s PASS\n" "$w"
            else
                printf "  WAIT=%-7s FAIL\n" "$w"
                FAILED=$((FAILED+1))
            fi
        done
        echo "== sweep: 桥 × 从机交叉矩阵 =="
        for br in 0 1; do
            for suite in regfile regfilegen; do
                TOTAL=$((TOTAL+1))
                if DUAL=$br bash "$0" $suite 2>&1 | grep -q "^PASS"; then
                    printf "  桥=%-8s + %-12s PASS\n" \
                           "$([ $br -eq 1 ] && echo 并发 || echo 串行)" "$suite"
                else
                    printf "  桥=%-8s + %-12s FAIL\n" \
                           "$([ $br -eq 1 ] && echo 并发 || echo 串行)" "$suite"
                    FAILED=$((FAILED+1))
                fi
            done
        done
        echo "== sweep: 延迟公式验证（延迟 = 7 + WAIT）=="
        for w in 0 1 2 4 8 16; do
            TOTAL=$((TOTAL+1))
            out=$(WAIT=$w bash "$0" latency 2>&1)
            expect=$((7+w))
            if echo "$out" | grep -q "^PASS" && echo "$out" | grep -q "写 $expect 周期"; then
                printf "  WAIT=%-3s → 延迟 %2d 周期  PASS\n" "$w" "$expect"
            else
                printf "  WAIT=%-3s → 延迟不符（期望 %d） FAIL\n" "$w" "$expect"
                FAILED=$((FAILED+1))
            fi
        done
        echo "------------------------------------------------------"
        if [ "$FAILED" -eq 0 ]; then
            echo "PASS: sweep —— $TOTAL 组参数组合全部通过"
            exit 0
        else
            echo "FAIL: sweep —— $FAILED/$TOTAL 组失败"
            exit 1
        fi
        ;;
    mutate)
        # 变异测试：验证测试能否发现人为注入的 RTL 缺陷
        ( cd axi && python3 mutate.py )
        ;;
    docs)
        # 生成所有 HTML 可视化文档（从源码解析，非手绘）
        echo "== docs: FSM 状态转移图 =="
        ( cd axi && python3 gen_fsm_html.py 2>&1 | tail -4 )
        echo "== docs: 接口逻辑映射 =="
        ( cd axi && python3 gen_interface_map.py 2>&1 | tail -3 )
        echo "== docs: 测试方案与覆盖缺口 =="
        ( cd axi && python3 gen_test_plan.py 2>&1 | tail -3 )
        echo "== docs: 寄存器手册（从 spec 生成）=="
        ( cd axi && python3 gen_regfile.py 2>&1 | tail -4 )
        echo "== docs: 门户索引 =="
        ( cd axi && python3 gen_portal.py 2>&1 | tail -8 )
        echo "== docs: 完成 =="
        echo "  axi/INDEX.html            ← 门户（从这里开始）"
        echo "  axi/DESIGN.html           设计文档（手写）"
        echo "  axi/FSM.html              FSM 状态转移图"
        echo "  axi/INTERFACE_MAP.html    接口逻辑映射"
        echo "  axi/TEST_PLAN.html        测试方案与覆盖缺口"
        echo "  axi/REGFILE.html          寄存器手册（从 regfile_spec.json 生成）"
        echo "  axi/COVERAGE.html         覆盖率可视化（需先跑 sim.sh coverage）"
        ;;
    wave)
        ( cd axi && build_run wave tb_axi_apb_sv "-I./sv -I. --trace" \
            axi_defs_pkg.sv apb_defs_pkg.sv sv/bus_txn_pkg.sv \
            sv/axi_lite_if.sv sv/apb_if.sv sv/axi_apb_bridge_sv.sv \
            sv/apb_slave_regs_sv.sv sv/axi_apb_monitor.sv \
            sv/axi_txn_host.sv sv/tb_axi_apb_sv.sv
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
        bash "$0" covgaps
        bash "$0" monitor
        bash "$0" monitorrules
        bash "$0" latency
        bash "$0" fairness
        bash "$0" params
        bash "$0" sva
        bash "$0" lint
        bash "$0" duallegal
        bash "$0" fieldiso
        bash "$0" dualerr
        bash "$0" resetdom
        bash "$0" handshake
        bash "$0" uvmlite
        bash "$0" asserteff
        bash "$0" dw64
        bash "$0" stress
        bash "$0" regfile
        bash "$0" regfilegen
        bash "$0" vip
        echo "################ 全部完成 ################"
        ;;
    *)
        echo "未知目标: $1"
        echo "可用: counter counter8 axi sv gaps dual stress covgaps"
        echo "      regfile regfilegen vip"
        echo "      lint duallegal fieldiso dualerr resetdom handshake monitor monitorrules latency fairness params sva
      uvmlite asserteff dw64"
        echo "      coverage mutate sweep docs wave all"
        exit 1
        ;;
esac
