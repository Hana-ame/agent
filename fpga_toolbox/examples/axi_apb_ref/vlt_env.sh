# ============================================================================
# vlt_env.sh —— Verilator 位置探测（被其他脚本 source）
# ============================================================================
# 背景（真实踩坑）：
#   原实现所有脚本都硬编码 VERILATOR_ROOT=/tmp/verilator，
#   而 setup_env.sh 默认也把 Verilator 编译安装到 /tmp/verilator。
#   /tmp 是易失目录：系统重启 / systemd-tmpfiles / 手工清理都会清空它。
#   一旦被清，7 个脚本同时失效，CI 直接跑不起来，且报错信息只是
#   「找不到 verilator」，看不出根因是安装位置选了易失目录。
#
#   本文件把探测顺序固定下来，并优先持久位置：
#     1) 显式 $VERILATOR（最高优先，用户明确指定）
#     2) $VERILATOR_ROOT（显式或已 export）
#     3) $HOME/.local/verilator          ← setup_env.sh 现在的默认安装位置
#     4) /tmp/verilator                  ← 兼容旧安装（会给出「易失」提示）
#     5) PATH 里的 verilator
#   找到后统一 export VERILATOR / VERILATOR_ROOT / VERILATOR_BIN / VLT_COV_DIR。
#
# 用法：
#   . "$(dirname "$0")/vlt_env.sh"    # 之后直接用 $VLT
#   返回码 0=找到  1=找不到
# ============================================================================

vlt_probe() {
    # 允许调用方已经设好 VERILATOR
    if [ -n "${VERILATOR:-}" ] && [ -x "${VERILATOR}" ]; then
        VLT="${VERILATOR}"
    elif [ -n "${VERILATOR_ROOT:-}" ] && [ -x "${VERILATOR_ROOT}/bin/verilator_bin" ]; then
        VLT="${VERILATOR_ROOT}/bin/verilator_bin"
    elif [ -x "${HOME}/.local/verilator/bin/verilator_bin" ]; then
        VLT="${HOME}/.local/verilator/bin/verilator_bin"
        VERILATOR_ROOT="${HOME}/.local/verilator/share/verilator"
        export VERILATOR_ROOT
    elif [ -x "/tmp/verilator/bin/verilator_bin" ]; then
        VLT="/tmp/verilator/bin/verilator_bin"
        VERILATOR_ROOT="/tmp/verilator"
        export VERILATOR_ROOT
        echo "WARN: 正在使用 /tmp/verilator —— /tmp 是易失目录，重启后会丢失。" >&2
        echo "      建议重新运行 bash setup_env.sh（装到 ~/.local/verilator）。" >&2
    elif command -v verilator >/dev/null 2>&1; then
        VLT="$(command -v verilator)"
    else
        return 1
    fi
    export VERILATOR="$VLT"

    # verilator_coverage 通常在 verilator 同目录
    local d
    d="$(dirname "$VLT")"
    for d in "$(dirname "$VLT")" "${VERILATOR_ROOT:-/nonexistent}/bin"; do
        if [ -x "$d/verilator_coverage" ]; then
            export VERILATOR_COV="$d/verilator_coverage"
            break
        fi
    done
    if [ -z "${VERILATOR_COV:-}" ] && command -v verilator_coverage >/dev/null 2>&1; then
        export VERILATOR_COV="$(command -v verilator_coverage)"
    fi
    return 0
}

# 版本解析（供调用方复用）
vlt_version() {
    local s
    s=$("${1:-$VLT}" --version 2>&1 | head -1 || true)
    echo "$s" | awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+\.[0-9]+$/) {print $i; exit}}'
}

# 版本是否满足最低要求（默认 5.040）：5.020 有跨文件虚接口缺陷会静默挂死
vlt_ok() {
    local ver
    ver="$(vlt_version "${1:-$VLT}")"
    [ -n "$ver" ] || return 1
    awk -v v="$ver" -v maj="${2:-5}" -v min="${3:-40}" 'BEGIN{
        split(v,a,".");
        if (a[1] > maj) print 1;
        else if (a[1] == maj && a[2] >= min) print 1;
        else print 0;
    }' | grep -q 1
}
