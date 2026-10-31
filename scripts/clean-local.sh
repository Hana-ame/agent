#!/usr/bin/env bash
# 清理本仓库的「本地可再生数据」。
#
# 默认只列出（dry-run），不会删除任何东西；确认无误后用 --yes 执行。
# 永远不会碰：models.json、.workbuddy/、taskpad/data/store.json、
#             taskpad/data/artifacts/、reports/、.git/
#
# 用法：
#   scripts/clean-local.sh                     # 列出全部目标与体积（不删）
#   scripts/clean-local.sh safe --yes          # 只清 Python 构建缓存
#   scripts/clean-local.sh bootstrap v8data --yes
#   scripts/clean-local.sh all --yes           # 清全部目标（含大目录，慎用）
#
# 目标：
#   safe          __pycache__ / .pytest_cache / *.egg-info（可再生，推荐）
#   bootstrap     bootstrap_runs/（v6 自举历史产物，约 86M）
#   v8data        v8/data/（dag8 SQLite 运行轨迹，约 34M）
#   screenshots   browser-screenshots/（本地截图，约 4.6M）
#   venv          .venv/（可重建的测试环境，约 32M）
#   scratch       根目录一次性脚本与模型快照（6 个文件）
#   taskpad-logs  taskpad/data 的日志与 store 备份（保留 store.json 与 artifacts）

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)

[ -d "$ROOT/.git" ] || { echo "✗ 不在仓库根目录（找不到 .git）：$ROOT" >&2; exit 1; }

ALL_TARGETS=(safe bootstrap v8data screenshots venv scratch taskpad-logs)
ASSUME_YES=0
WANT_ALL=0
SELECTED=()

usage() { sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; }

for arg in "$@"; do
  case "$arg" in
    --yes|-y)   ASSUME_YES=1 ;;
    --dry-run)  ASSUME_YES=0 ;;
    --all)      WANT_ALL=1 ;;
    -h|--help)  usage; exit 0 ;;
    safe|bootstrap|v8data|screenshots|venv|scratch|taskpad-logs) SELECTED+=("$arg") ;;
    *) echo "✗ 未知参数：$arg（--help 查看用法）" >&2; exit 2 ;;
  esac
done

[ "$WANT_ALL" = 1 ] && SELECTED=("${ALL_TARGETS[@]}")

LIST_ONLY=0
if [ "${#SELECTED[@]}" -eq 0 ]; then
  LIST_ONLY=1
  SELECTED=("${ALL_TARGETS[@]}")
fi

if [ "$ASSUME_YES" = 1 ] && [ "$LIST_ONLY" = 1 ]; then
  echo "✗ 拒绝执行：--yes 需要显式列出目标，或使用 --all" >&2
  exit 2
fi

# 安全目标：排除 .venv 与 bootstrap_runs（它们由独立目标负责）
safe_paths() {
  find "$ROOT" \
    -path "$ROOT/.venv" -prune -o \
    -path "$ROOT/.git" -prune -o \
    -path "$ROOT/bootstrap_runs" -prune -o \
    \( -name __pycache__ -o -name .pytest_cache -o -name '*.egg-info' \) -print
}

target_paths() {
  case "$1" in
    safe)        safe_paths ;;
    bootstrap)   [ -e "$ROOT/bootstrap_runs" ] && echo "$ROOT/bootstrap_runs" ;;
    v8data)      [ -e "$ROOT/v8/data" ] && echo "$ROOT/v8/data" ;;
    screenshots) [ -e "$ROOT/browser-screenshots" ] && echo "$ROOT/browser-screenshots" ;;
    venv)        [ -e "$ROOT/.venv" ] && echo "$ROOT/.venv" ;;
    scratch)     for f in .fix_example.py .gen_models.py .probe_cap.py .verify_cap.py .models_new.b64 .models_new.json; do
                   [ -e "$ROOT/$f" ] && echo "$ROOT/$f"
                 done ;;
    taskpad-logs) for f in "$ROOT"/taskpad/data/*.log "$ROOT"/taskpad/data/store.json.bak-*; do
                    [ -e "$f" ] && echo "$f"
                  done ;;
  esac
}

human_kb() {
  awk -v kb="$1" 'BEGIN {
    split("KB MB GB TB", u, " "); i = 1;
    while (kb >= 1024 && i < 4) { kb /= 1024; i++ }
    printf (i == 1 ? "%d%s" : "%.1f%s"), kb, u[i]
  }'
}

echo "仓库：$ROOT"
if [ "$ASSUME_YES" = 1 ]; then
  echo "模式：执行删除（--yes）"
else
  echo "模式：dry-run（只列不删；加 --yes 才执行）"
fi
echo

total_kb=0
for target in "${SELECTED[@]}"; do
  mapfile -t paths < <(target_paths "$target")
  if [ "${#paths[@]}" -eq 0 ]; then
    printf '  %-13s 已不存在，跳过\n' "$target"
    continue
  fi
  for p in "${paths[@]}"; do
    case "$p" in
      "$ROOT"/*) ;;
      *) echo "✗ 拒绝：路径不在仓库内：$p" >&2; exit 1 ;;
    esac
  done
  kb=$(du -skc "${paths[@]}" 2>/dev/null | tail -1 | cut -f1)
  kb=${kb:-0}
  total_kb=$((total_kb + kb))
  printf '  %-13s %3d 项  %8s\n' "$target" "${#paths[@]}" "$(human_kb "$kb")"
  if [ "$ASSUME_YES" = 0 ]; then
    printf '      %s\n' "${paths[@]}"
  fi
done

echo
echo "合计约 $(human_kb "$total_kb")"

if [ "$ASSUME_YES" = 0 ]; then
  echo
  echo "未删除任何文件。确认后执行：scripts/clean-local.sh ${SELECTED[*]} --yes"
  exit 0
fi

removed=0
for target in "${SELECTED[@]}"; do
  mapfile -t paths < <(target_paths "$target")
  for p in "${paths[@]}"; do
    case "$p" in
      "$ROOT"/*) rm -rf -- "$p" && removed=$((removed + 1)) ;;
    esac
  done
done
echo "✓ 已删除 $removed 项。"