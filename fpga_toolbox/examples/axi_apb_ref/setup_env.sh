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
#   bash setup_env.sh                   # 默认装到 ~/.local/verilator（持久）
#   PREFIX=$HOME/tools bash setup_env.sh  # 自定义安装位置
#
# 注意：旧版本默认装到 /tmp/verilator。但 /tmp 是易失目录，
# 系统重启 / tmpfiles 清理会让 7 个脚本同时失效。因此默认位置已改为
# ~/.local/verilator；/tmp/verilator 仅作为兼容回退被探测。
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
PREFIX="${PREFIX:-$HOME/.local/verilator}"
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

# 注意：即使 Verilator 已就绪，也要继续检查 slang，
# 不能提前 exit（否则「环境已就绪」是假象）。
VLT_READY=0
if [ -n "$FOUND" ]; then
    VLT_READY=1
    echo -e "  ${D}推荐：export VERILATOR_ROOT=$(dirname "$(dirname "$FOUND")")${R}"
fi

if [ "$VLT_READY" -eq 0 ] && [ "$CHECK_ONLY" -eq 1 ]; then
    echo ""
    printf "${Y}✗ 未找到 >= %d.%03d 的 Verilator${R}\n" "$MIN_VER_MAJOR" "$MIN_VER_MINOR"
    echo -e "  运行 ${B}bash setup_env.sh${R} 自动安装"
    exit 1
fi

# ---- 2. 构建依赖检查（仅 Verilator 未就绪时需要）----
if [ "$VLT_READY" -eq 1 ]; then
    SKIP_VLT_BUILD=1
else
    SKIP_VLT_BUILD=0
fi
if [ "$SKIP_VLT_BUILD" -eq 0 ]; then
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
# 源码目录与安装目录必须分开：旧实现 SRC_DIR="$PREFIX"，
# 源码和安装树混在同一目录，既污染 prefix，也让 rm -rf 非常危险。
echo ""
echo -e "${B}获取 Verilator 源码${R}"
SRC_DIR="${SRC_DIR:-${TMPDIR:-/tmp}/verilator-src}"
# 固定 tag：旧实现直接 clone master（devel），版本随时漂移且
# 曾出现按不存在的 tag（v5.055 未发布）clone 直接失败。
VLT_TAG="${VLT_TAG:-v5.054}"
if [ -d "$SRC_DIR/.git" ]; then
    echo -e "  ${D}已有源码目录，复用：$SRC_DIR${R}"
    ( cd "$SRC_DIR" && git fetch --depth 1 origin         "+refs/tags/$VLT_TAG:refs/tags/$VLT_TAG" 2>/dev/null || true )
    ( cd "$SRC_DIR" && git checkout -q "$VLT_TAG" 2>/dev/null || true )
else
    echo -e "  ${D}克隆 $VLT_TAG 到 $SRC_DIR（浅克隆，约 1-2 分钟）${R}"
    rm -rf "$SRC_DIR"
    if ! git clone --depth 1 --branch "$VLT_TAG" \
            https://github.com/verilator/verilator.git "$SRC_DIR"; then
        echo -e "  ${Y}✗ clone 失败（网络问题？）${R}"
        echo "  手动方式："
        echo "    git clone --depth 1 --branch $VLT_TAG https://github.com/verilator/verilator.git $SRC_DIR"
        echo "    cd $SRC_DIR && autoconf && ./configure --prefix=$PREFIX"
        echo '    make -j$(nproc)'
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

echo -e "  ${D}configure --prefix=$PREFIX${R}"
if ! ./configure --prefix="$PREFIX" >/tmp/verilator_configure.log 2>&1; then
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
# make install 会为 man page 调 help2man；该工具缺失时 install 以 2 退出，
# 但二进制其实已经构建好了（Error 127 = command not found）。
# 这里不把「man page 生成失败」当成安装失败，改为手工复制兜底，
# 否则缺 help2man 的机器永远装不上（本项目实际踩到过）。
if ! make install >/tmp/verilator_install.log 2>&1; then
    if [ -x bin/verilator_bin ]; then
        echo -e "  ${D}make install 失败（多为缺 help2man），改用手工安装${R}"
        mkdir -p "$PREFIX/bin" "$PREFIX/share/verilator/include" "$PREFIX/share/verilator/bin"
        cp bin/verilator_bin "$PREFIX/bin/"
        [ -x bin/verilator_bin_dbg ] && cp bin/verilator_bin_dbg "$PREFIX/bin/"
        [ -x bin/verilator ] && cp bin/verilator "$PREFIX/bin/"
        cp -r include/* "$PREFIX/share/verilator/include/" 2>/dev/null || true
        # verilator_includer 是 __ALL.cpp 生成所必需的 python 脚本，
        # 漏装会让所有构建在 make 阶段报 can't open file（曾实际踩到）
        cp bin/verilator_includer "$PREFIX/share/verilator/bin/" 2>/dev/null || true
        for _x in verilator_coverage verilator_gantt verilator_profcfunc; do
            [ -f "bin/$_x" ] && cp "bin/$_x" "$PREFIX/bin/" && chmod +x "$PREFIX/bin/$_x"
        done
        # 生成可用的 verilator 驱动脚本（verilator_bin 需要 VERILATOR_ROOT）
        cat > "$PREFIX/bin/verilator" <<'WRAP'
#!/bin/sh
: "${VERILATOR_ROOT:=@PREFIX@/share/verilator}"
export VERILATOR_ROOT
exec "@PREFIX@/bin/verilator_bin" "$@"
WRAP
        sed -i "s|@PREFIX@|$PREFIX|g" "$PREFIX/bin/verilator"
        chmod +x "$PREFIX/bin/verilator"
    else
        echo -e "  ${Y}✗ make install 失败${R}"
        tail -8 /tmp/verilator_install.log | sed 's/^/      /'
        exit 1
    fi
fi

# ---- 5. 验证 ----
echo ""
echo -e "${B}验证安装${R}"
NEW_V=$(check_one "$PREFIX/bin/verilator_bin") || NEW_V=""
if [ -z "$NEW_V" ] || ! ver_ok "$NEW_V"; then
    echo -e "  ${Y}✗ 安装后仍找不到可用版本${R}"
    exit 1
fi
echo -e "  ${G}✓${R} Verilator $NEW_V 安装成功"

echo ""
echo -e "${G}✓${R} Verilator 准备完成"
fi  # end SKIP_VLT_BUILD

# ============================================================================
# slang（编译检查工具）
# ============================================================================
# 为什么需要：ci.sh 的 G1 用 slang 做 19 项编译检查。
# 没有 slang 时 G1 会判 FAIL（不再静默 SKIP）。
# slang 是单一静态二进制，安装比 Verilator 简单。
# ============================================================================
echo ""
echo -e "${B}环境检查：slang（SV 编译检查）${R}"

SLANG_BIN=$(command -v slang 2>/dev/null || true)
if [ -n "$SLANG_BIN" ]; then
    SV=$( "$SLANG_BIN" --version 2>&1 | head -1 || true )
    echo -e "  ${G}✓${R} 已安装: $SLANG_BIN（$SV）"
else
    echo -e "  ${Y}✗${R} 未找到 slang"
    if [ "$CHECK_ONLY" -eq 1 ]; then
        echo -e "  ${D}运行 bash setup_env.sh 自动安装${R}"
        exit 1
    fi

    SLANG_VER="${SLANG_VERSION:-v10.0}"
    SLANG_URL="https://github.com/MikePopoloski/slang/releases/download/${SLANG_VER}/slang-${SLANG_VER#v}-linux-x86_64.tar.gz"
    echo -e "  ${D}尝试从 GitHub 下载 ${SLANG_VER}...${R}"

    TMPD=$(mktemp -d)
    if command -v curl >/dev/null 2>&1; then
        DL="curl -fsSL"
    elif command -v wget >/dev/null 2>&1; then
        DL="wget -qO-"
    else
        DL=""
    fi

    INSTALLED=0
    if [ -n "$DL" ]; then
        if $DL "$SLANG_URL" -o "$TMPD/slang.tar.gz" 2>/dev/null \
           || $DL "$SLANG_URL" > "$TMPD/slang.tar.gz" 2>/dev/null; then
            if tar -xzf "$TMPD/slang.tar.gz" -C "$TMPD" 2>/dev/null; then
                SLANG_FOUND=$(find "$TMPD" -name slang -type f -perm -u+x 2>/dev/null | head -1)
                if [ -n "$SLANG_FOUND" ]; then
                    mkdir -p "$HOME/.local/bin"
                    cp "$SLANG_FOUND" "$HOME/.local/bin/slang"
                    chmod +x "$HOME/.local/bin/slang"
                    echo -e "  ${G}✓${R} 已安装到 $HOME/.local/bin/slang"
                    echo -e "  ${D}  请确保 $HOME/.local/bin 在 PATH 中${R}"
                    INSTALLED=1
                fi
            fi
        fi
    fi
    rm -rf "$TMPD"

    if [ "$INSTALLED" -eq 0 ]; then
        echo -e "  ${Y}✗ 自动安装失败（可能无网络或版本号不对）${R}"
        echo "  手动安装方式（任选其一）："
        echo "    a) 下载预编译包："
        echo "       https://github.com/MikePopoloski/slang/releases"
        echo "       解压后把 slang 放到 PATH（如 ~/.local/bin/）"
        echo "    b) 从源码构建（需 cmake + C++20 编译器）："
        echo "       git clone https://github.com/MikePopoloski/slang.git"
        echo "       cd slang && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
        exit 1
    fi
fi

echo ""
echo -e "${G}${B}环境准备完成${R}"
echo ""
echo "  验证："
echo -e "    ${D}bash ci.sh --stage 1     # 编译检查${R}"
echo -e "    ${D}bash sim.sh counter      # 简单仿真${R}"
