#!/usr/bin/env bash
# ============================================================================
# setup_env.sh —— 环境准备（自动获取/编译 Verilator 5.040+）
# ============================================================================
# 为什么需要这个脚本
#   本项目统一用 Verilator（仿真 + 覆盖率），且要求 **>= 5.040**：
#   apt 自带的 5.020 有跨文件虚接口缺陷，仿真是「静默挂死」
#   （无输出、无报错、进程不动），比直接报错难排查得多。
#
#   而 5.040+ 通常不在发行版仓库里，需要自编译。
#   自编译步骤不复杂但容易漏依赖，所以做成脚本。
#
# 用法
#   bash setup_env.sh                   # 默认装到 /tmp/verilator
#   PREFIX=$HOME/tools bash setup_env.sh  # 自定义安装位置
#   bash setup_env.sh --check           # 只检查现有环境是否满足
#
# 幂等：已满足要求时直接退出，不重复编译。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1

if [ -t 1 ]; then R='\033[0m'; G='\033[32m'; Y='\033[31m'; B='\033[1m'; D='\033[2m'
else R=''; G=''; Y=''; B=''; D=''; fi

MIN_VER_MAJOR=5
MIN_VER_MINOR=40
PREFIX="${PREFIX:-/tmp/verilator}"
CHECK_ONLY=0
for a in "$@"; do
    case "$a" in
        --check) CHECK_ONLY=1 ;;
        --help|-h) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    esac
done

# ---- 版本工具 ----
parse_ver() {  # 从 "Verilator 5.055 devel ..." 取 X.Y
    echo "$1" | awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+\.[0-9]+$/) {print $i; exit}}'
}

ver_ok() {  # ver_ok 5.055
    awk -v v="$1" -v M="$MIN_VER_MAJOR" -v m="$MIN_VER_MINOR" 'BEGIN{
        split(v, a, ".");
        if (a[1] > M) exit 0;
        if (a[1] == M && a[2] >= m) exit 0;
        exit 1;
    }'
}

check_one() {  # check_one <path> → 输出版本或空
    [ -x "$1" ] || return 1
    # 注意：不能写 "$("$1" --version)"，bash 无法解析这种嵌套双引号
    local raw
    raw=$("$1" --version 2>&1 | head -1 || true)
    parse_ver "$raw"
}

printf "${B}环境检查：Verilator >= %d.%03d${R}\n" "$MIN_VER_MAJOR" "$MIN_VER_MINOR"

# ---- 1. 检查已有环境 ----
FOUND=""
for cand in "${VERILATOR:-}" \
            "${VERILATOR_ROOT:-}/bin/verilator_bin" \
            "$PREFIX/bin/verilator_bin" \
            "$(command -v verilator 2>/dev/null || true)"; do
    [ -z "$cand" ] && continue
    v=$(check_one "$cand") || continue
    if ver_ok "$v"; then
        echo -e "  ${G}✓${R} 找到可用版本: $cand（Verilator $v）"
        FOUND="$cand"
        break
    else
        echo -e "  ${D}- 跳过 $cand（Verilator ${v:-未知}，低于要求）${R}"
    fi
done

if [ -n "$FOUND" ]; then
    echo ""
    echo -e "${G}${B}环境已就绪${R}"
    echo -e "  推荐用法：${D}export VERILATOR_ROOT=$(dirname "$(dirname "$FOUND")")${R}"
    exit 0
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
    echo ""
    printf "${Y}✗ 未找到 >= %d.%03d 的 Verilator${R}\n" "$MIN_VER_MAJOR" "$MIN_VER_MINOR"
    echo -e "  运行 ${B}bash setup_env.sh${R} 自动安装"
    exit 1
fi

# ---- 2. 构建依赖检查 ----
echo ""
echo -e "${B}检查构建依赖${R}"
MISSING=()
for tool in git make autoconf g++ flex bison; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo -e "  ${G}✓${R} $tool"
    else
        echo -e "  ${Y}✗${R} $tool 缺失"
        MISSING+=("$tool")
    fi
done
# flex/bison 在某些系统叫 flex/bison，也可能缺 libfl
if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo -e "${Y}缺少构建依赖：${MISSING[*]}${R}"
    echo "  请先安装（Ubuntu/Debian 示例）："
    echo "    sudo apt install git make autoconf g++ flex bison libfl-dev help2man"
    exit 1
fi

# ---- 3. 获取源码 ----
echo ""
echo -e "${B}获取 Verilator 源码${R}"
SRC_DIR="$PREFIX"
if [ -d "$SRC_DIR/.git" ]; then
    echo -e "  ${D}已有源码目录，复用：$SRC_DIR${R}"
    ( cd "$SRC_DIR" && git fetch --depth 1 origin 2>/dev/null || true )
else
    echo -e "  ${D}克隆到 $SRC_DIR（浅克隆，约 1-2 分钟）${R}"
    rm -rf "$SRC_DIR"
    if ! git clone --depth 1 https://github.com/verilator/verilator.git "$SRC_DIR"; then
        echo -e "  ${Y}✗ clone 失败（网络问题？）${R}"
        echo "  手动方式："
        echo "    git clone --depth 1 https://github.com/verilator/verilator.git $SRC_DIR"
        echo "    cd $SRC_DIR && autoconf && ./configure --prefix=$SRC_DIR"
        echo '    make -j$(nproc) && make install'
        exit 1
    fi
fi

# ---- 4. 编译 ----
echo ""
echo -e "${B}编译安装（约 5-15 分钟）${R}"
cd "$SRC_DIR" || exit 1
JOBS=$(nproc 2>/dev/null || echo 4)

if [ ! -f configure ]; then
    echo -e "  ${D}生成 configure${R}"
    autoconf >/dev/null 2>&1 || { echo -e "  ${Y}✗ autoconf 失败${R}"; exit 1; }
fi

echo -e "  ${D}configure --prefix=$SRC_DIR${R}"
if ! ./configure --prefix="$SRC_DIR" >/tmp/verilator_configure.log 2>&1; then
    echo -e "  ${Y}✗ configure 失败，详见 /tmp/verilator_configure.log${R}"
    tail -8 /tmp/verilator_configure.log | sed 's/^/      /'
    exit 1
fi

echo -e "  ${D}make -j$JOBS（耗时较长，请稍候）${R}"
if ! make -j"$JOBS" >/tmp/verilator_make.log 2>&1; then
    echo -e "  ${Y}✗ make 失败，详见 /tmp/verilator_make.log${R}"
    tail -8 /tmp/verilator_make.log | sed 's/^/      /'
    exit 1
fi

echo -e "  ${D}make install${R}"
if ! make install >/tmp/verilator_install.log 2>&1; then
    echo -e "  ${Y}✗ make install 失败${R}"
    exit 1
fi

# ---- 5. 验证 ----
echo ""
echo -e "${B}验证安装${R}"
NEW_V=$(check_one "$SRC_DIR/bin/verilator_bin") || NEW_V=""
if [ -z "$NEW_V" ] || ! ver_ok "$NEW_V"; then
    echo -e "  ${Y}✗ 安装后仍找不到可用版本${R}"
    exit 1
fi
echo -e "  ${G}✓${R} Verilator $NEW_V 安装成功"

echo ""
echo -e "${G}${B}环境准备完成${R}"
echo ""
echo "  后续使用（任选其一）："
echo -e "    ${B}export VERILATOR_ROOT=$SRC_DIR${R}   # 推荐，sim.sh 会自动识别"
echo -e "    ${B}export VERILATOR=$SRC_DIR/bin/verilator_bin${R}"
echo ""
echo "  验证："
echo -e "    ${D}bash sim.sh counter${R}"
