#!/usr/bin/env bash
# ============================================================================
# repro.sh —— 复现性验证：干净 clone 后项目能否直接跑通
# ============================================================================
# 背景：
#   本项目曾出现「本地能跑、干净 clone 跑不通」的问题：
#     - axi/run 目录只在顶层创建，子目录首次运行报 Can't write file
#     - 手改生成物（tb_regfile_gen.sv 的 DUAL 参数）被生成器覆盖
#   这类问题在本地开发时完全不可见（因为本地已有残留状态），
#   只有「从零 clone」才能暴露。
#
#   本脚本把「干净 clone → 全量回归」做成可重复的检查。
#
# 用法:
#   bash repro.sh              # 完整验证（clone + 全量回归，约 5 分钟）
#   bash repro.sh --quick      # 只跑 4 个基础套件（约 1 分钟）
#   bash repro.sh --keep       # 保留临时 clone 目录（便于排查）
#
# 退出码: 0 = 复现成功，1 = 失败
# ============================================================================
set -uo pipefail

REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)
if [ -z "$REPO_ROOT" ]; then
    echo "错误：不在 git 仓库内" >&2
    exit 1
fi
BRANCH=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)

if [ -t 1 ]; then R='\033[0m'; G='\033[32m'; Y='\033[31m'; B='\033[1m'; D='\033[2m'
else R=''; G=''; Y=''; B=''; D=''; fi

QUICK=0
KEEP=0
for a in "$@"; do
    case "$a" in
        --quick) QUICK=1 ;;
        --keep)  KEEP=1 ;;
        --help|-h) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    esac
done

WORK=$(mktemp -d /tmp/repro.XXXXXX)
CLONE="$WORK/repo"
FAIL=0

cleanup() {
    if [ "$KEEP" -eq 1 ]; then
        echo -e "${D}临时目录保留: $WORK${R}"
    else
        rm -rf "$WORK"
    fi
}
trap cleanup EXIT

echo -e "${B}复现性验证${R}"
echo -e "${D}源仓库: $REPO_ROOT (分支 $BRANCH)${R}"
echo -e "${D}临时目录: $CLONE${R}"
echo ""

# ---- 1. 干净 clone（只取该分支，不含工作区残留）----
echo -e "${B}[1/4] 干净 clone${R}"
if ! git clone -q --branch "$BRANCH" --single-branch "$REPO_ROOT" "$CLONE" 2>/dev/null; then
    echo -e "  ${Y}✗ clone 失败${R}"
    exit 1
fi
cd "$CLONE" || exit 1
echo "  ✓ clone 完成（$(git rev-parse --short HEAD)）"

# ---- 2. 检查工作区是否真的干净（无未提交/未跟踪的必需文件）----
echo -e "${B}[2/4] 检查文件完整性${R}"
DIRTY=$(git status --porcelain | wc -l | tr -d ' ')
if [ "$DIRTY" -ne 0 ]; then
    echo -e "  ${Y}✗ clone 后工作区不干净（$DIRTY 项）${R}"
    FAIL=1
else
    echo "  ✓ 工作区干净"
fi

# 关键文件存在性检查（这些曾因被 gitignore 或未提交而缺失）
REQUIRED=(
    "verilog/sim.sh"
    "verilog/ci.sh"
    "verilog/audit.sh"
    "verilog/axi/sv/apb_slave_regfile_gen.sv"
    "verilog/axi/sv/tb_regfile_gen.sv"
    "verilog/axi/regfile_spec.json"
)
MISSING=0
for f in "${REQUIRED[@]}"; do
    if [ ! -f "$f" ]; then
        echo -e "  ${Y}✗ 缺少必需文件: $f${R}"
        MISSING=$((MISSING+1))
    fi
done
if [ "$MISSING" -eq 0 ]; then
    echo "  ✓ ${#REQUIRED[@]} 个关键文件齐全"
else
    FAIL=1
fi

# ---- 3. 生成物一致性：重新生成后应无差异 ----
echo -e "${B}[3/4] 生成物一致性${R}"
cd "$CLONE/verilog/axi" || exit 1
if [ -f gen_regfile.py ]; then
    python3 gen_regfile.py >/dev/null 2>&1
    cd "$CLONE" || exit 1
    GEN_DIFF=$(git status --porcelain -- verilog/axi/sv/apb_slave_regfile_gen.sv \
                                           verilog/axi/sv/tb_regfile_gen.sv | wc -l | tr -d ' ')
    if [ "$GEN_DIFF" -eq 0 ]; then
        echo "  ✓ 重新生成后无差异（生成器与提交的产物一致）"
    else
        echo -e "  ${Y}✗ 重新生成后产物有差异（提交的生成物已过期或被手改）${R}"
        git status --porcelain -- verilog/axi/sv/ | sed 's/^/      /'
        FAIL=1
    fi
else
    echo -e "  ${D}（无生成器，跳过）${R}"
fi

# ---- 4. 全量回归 ----
cd "$CLONE/verilog" || exit 1
if [ "$QUICK" -eq 1 ]; then
    echo -e "${B}[4/4] 快速回归（4 套件）${R}"
    SUITES="counter counter8 axi sv"
    EXPECT=4
else
    echo -e "${B}[4/4] 全量回归（$(sed -n '/^    all)/,/^        ;;/p' sim.sh | grep -cE 'bash "\$0"' | tr -d ' ' || echo 22) 项）${R}"
    SUITES="all"
    EXPECT=21
fi

if [ "$SUITES" = "all" ]; then
    OUT=$(timeout 1500 bash sim.sh all 2>&1)
else
    OUT=""
    for s in $SUITES; do
        OUT="$OUT$(timeout 300 bash sim.sh "$s" 2>&1)"$'\n'
    done
fi

PASSES=$(echo "$OUT" | grep -cE "^PASS|^VIP PASS" || true)
if [ "${PASSES:-0}" -ge "$EXPECT" ]; then
    echo "  ✓ $PASSES 个套件通过（期望 >= $EXPECT）"
else
    echo -e "  ${Y}✗ 仅 $PASSES 个套件通过（期望 >= $EXPECT）${R}"
    echo "$OUT" | grep -E "^FAIL|BUILD MISSING|COMPILE FAILED|%Error" | head -10 | sed 's/^/      /'
    FAIL=1
fi

# ---- 汇总 ----
echo ""
echo -e "${B}────────────────────────────────────────${R}"
if [ "$FAIL" -eq 0 ]; then
    echo -e "  ${G}${B}复现成功${R} —— 干净 clone 可直接跑通"
    exit 0
else
    echo -e "  ${Y}${B}复现失败${R} —— 存在本地状态依赖"
    echo -e "  ${D}提示：用 --keep 保留临时目录以便排查${R}"
    exit 1
fi
