#!/usr/bin/env bash
# ============================================================================
# scripts_check.sh —— 所有 shell 脚本的可用性检查
# ============================================================================
# 背景：
#   项目里有 7 个 shell 脚本，但部分长期未维护。
#   本轮发现 axi/run_sv_chain.sh 已失效（源文件列表缺 axi_apb_monitor.sv，
#   运行直接报 unknown module）。这类「文档说能用、实际不能用」的问题
#   不会触发任何测试，只有主动去跑才会暴露。
#
# 本脚本对每个脚本做静态 + 动态检查：
#   1. 语法检查（bash -n）
#   2. 引用的文件是否存在
#   3. 对可运行脚本做实际执行验证
#
# 用法:
#   bash scripts_check.sh          # 检查全部
#   bash scripts_check.sh --list   # 列出检查项
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1

if [ -t 1 ]; then R='\033[0m'; G='\033[32m'; Y='\033[31m'; B='\033[1m'; D='\033[2m'
else R=''; G=''; Y=''; B=''; D=''; fi

FAST=0
for a in "$@"; do
    case "$a" in
        --fast) FAST=1 ;;
        --help|-h) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    esac
done

PASS=0
FAIL=0
SKIP=0
declare -a RESULTS=()

record() {  # record <name> <PASS|FAIL|SKIP> <detail>
    case "$2" in
        PASS) PASS=$((PASS+1)) ;;
        FAIL) FAIL=$((FAIL+1)) ;;
        SKIP) SKIP=$((SKIP+1)) ;;
    esac
    RESULTS+=("$2|$1|$3")
}

# ---- 1. 语法检查 ----
check_syntax() {
    local f="$1"
    if bash -n "$f" 2>/dev/null; then
        record "$f 语法" PASS ""
    else
        record "$f 语法" FAIL "bash -n 失败"
    fi
}

# ---- 2. 引用文件存在性（针对硬编码的源文件列表）----
check_refs() {
    local f="$1"
    # 提取 .sv / .v 文件引用，检查是否存在于脚本所在目录树
    local missing=0
    local dir
    dir=$(dirname "$f")
    while IFS= read -r ref; do
        [ -z "$ref" ] && continue
        # 排除运行时动态生成的文件（如审计脚本在临时目录造的 TB）
        case "$ref" in
            work/*|/tmp/*|*t_hang*) continue ;;
        esac
        # 支持多种路径基准：脚本目录 / 当前目录 / 仓库根
        # 支持的路径基准（bare 文件名可能位于 sv/ 子目录）
        if [ -f "$dir/$ref" ]     || [ -f "$ref" ]         \
        || [ -f "axi/$ref" ]      || [ -f "axi/sv/$ref" ]  \
        || [ -f "../$ref" ]       || [ -f "../../$ref" ]   \
        || [ -f "../axi/sv/$ref" ] || [ -f "../../axi/sv/$ref" ]; then
            continue
        fi
        echo -e "  ${D}  引用缺失: $ref${R}"
        missing=$((missing+1))
    # 注意：排除 $VAR/xxx.sv 这类变量路径 —— 形如 $SV/foo.sv，
    # grep 会把 "SV/foo.sv" 当字面路径，永远找不到 → 误报。
    # 这是 audit.sh 在干净 clone 下报「2 处引用缺失」的原因。
    done < <(grep -oE '[A-Za-z0-9_/]+\.(sv|v)\b' "$f" 2>/dev/null \
             | grep -vE '^[A-Z][A-Z0-9_]*/' | sort -u)

    if [ "$missing" -eq 0 ]; then
        : # record 在调用处统一记
    fi
    return "$missing"
}

# ---- 3. 实际执行验证（针对本机可运行的脚本）----
run_script() {
    local f="$1" expect="${2:-0}" timeout_s="${3:-200}"
    local out rc
    out=$(cd "$(dirname "$f")" 2>/dev/null && timeout "$timeout_s" bash "$(basename "$f")" 2>&1)
    rc=$?
    if [ "$rc" -eq "$expect" ]; then
        record "$f 执行" PASS "exit=$rc"
    else
        record "$f 执行" FAIL "exit=$rc（期望 $expect）"
        echo "$out" | tail -4 | sed 's/^/      /'
    fi
}

echo -e "${B}Shell 脚本可用性检查${R}"
echo -e "${D}（语法 + 引用文件 + 实际执行）${R}"
echo ""

SCRIPTS=$(git ls-files '*.sh' 2>/dev/null || find . -name "*.sh" -not -path "./run/*")

# ---- 第一轮：语法 + 引用 ----
echo -e "${B}[1/3] 语法检查${R}"
for f in $SCRIPTS; do
    check_syntax "$f"
done
for r in "${RESULTS[@]}"; do
    st="${r%%|*}"; rest="${r#*|}"; nm="${rest%%|*}"
    [ "$st" = "PASS" ] && echo -e "  ${G}✓${R} $nm" || echo -e "  ${Y}✗${R} $nm"
done

echo ""
echo -e "${B}[2/3] 引用文件检查${R}"
for f in $SCRIPTS; do
    n=0
    check_refs "$f" || n=$?
    if [ "$n" -eq 0 ]; then
        echo -e "  ${G}✓${R} $f（引用完整）"
        record "$f 引用" PASS ""
    else
        echo -e "  ${Y}✗${R} $f（$n 处引用缺失）"
        record "$f 引用" FAIL "$n 缺失"
    fi
done

# ---- 第三轮：实际执行（只跑本机可用的）----
echo ""
echo -e "${B}[3/3] 实际执行${R}"

# 本机可运行的脚本（有明确权限/依赖）
# audit.sh 的前置检查要求 RTL 干净；此处用 --allow-dirty
# 避免「开发中工作区不干净」被误判为脚本失效
# audit.sh 首次执行耗时取决于机器负载：
#   完整审计含 A8（跑 G6 变异约 350s）+ A1-A7，总计可能 600s+
#   --fast 模式只验证它能启动（--list），避免 CI G1 被拖到 10 分钟
if [ "$FAST" -eq 1 ]; then
    out=$(timeout 120 bash audit.sh --list 2>&1); rc=$?
    if [ "$rc" -eq 0 ] && echo "$out" | grep -q "A1 "; then
        echo -e "  ${D}- 部分验证${R} audit.sh（--fast：仅验证可启动与审计项清单）"
        record "audit.sh 执行" SKIP "--fast"
    else
        echo -e "  ${Y}✗${R} audit.sh 无法启动（exit=$rc）"
        record "audit.sh 执行" FAIL "exit=$rc"
    fi
else
    # 完整审计：A8 会跑 G6 变异（~350s），默认 900s 超时
    out=$(timeout "${AUDIT_TIMEOUT:-900}" bash audit.sh --allow-dirty 2>&1); rc=$?
    if [ "$rc" -eq 0 ]; then
        record "audit.sh 执行" PASS "exit=0"
    else
        record "audit.sh 执行" FAIL "exit=$rc"
        echo "$out" | tail -4 | sed 's/^/      /'
    fi
fi
run_script "axi/lint_all.sh" 0 200 || true
if [ "$FAST" -eq 1 ]; then
    echo -e "  ${D}- 跳过${R} axi/run_sv_chain.sh 执行（--fast 模式）"
    record "axi/run_sv_chain.sh 执行" SKIP "--fast"
else
    run_script "axi/run_sv_chain.sh" 0 200 || true
fi
# repro.sh 执行（--fast 模式跳过，避免 CI G1 耗时过长）
if [ "$FAST" -eq 1 ]; then
    echo -e "  ${D}- 跳过${R} repro.sh 执行（--fast 模式）"
    record "repro.sh 执行" SKIP "--fast"
else
    out=$(timeout 300 bash repro.sh --quick 2>&1); rc=$?
    if [ "$rc" -eq 0 ]; then
        record "repro.sh 执行(--quick)" PASS "exit=0"
    else
        record "repro.sh 执行(--quick)" FAIL "exit=$rc"
        echo "$out" | tail -4 | sed 's/^/      /'
    fi
fi

# ---- 汇总 ----
echo ""
echo -e "${B}────────────────────────────────────────${R}"
for r in "${RESULTS[@]}"; do
    st="${r%%|*}"; rest="${r#*|}"; nm="${rest%%|*}"; dt="${rest#*|}"
    case "$st" in
        PASS) printf "  ${G}✓ PASS${R}  %-40s ${D}%s${R}\n" "$nm" "$dt" ;;
        SKIP) printf "  ${D}- SKIP${R}  %-40s ${D}%s${R}\n" "$nm" "$dt" ;;
        *)    printf "  ${Y}✗ FAIL${R}  %-40s ${D}%s${R}\n" "$nm" "$dt" ;;
    esac
done
echo ""
if [ "$FAIL" -eq 0 ]; then
    echo -e "  ${G}${B}全部脚本可用${R}（PASS=$PASS SKIP=$SKIP）"
    exit 0
else
    echo -e "  ${Y}${B}$FAIL 项检查失败${R}（PASS=$PASS SKIP=$SKIP）"
    exit 1
fi
