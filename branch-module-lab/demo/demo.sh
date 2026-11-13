#!/usr/bin/env bash
# 端到端演示：branch（单一功能）→ merge（成为模块）→ 更新（自动传播到模块 + 运行时热更新）
#
# 机制：
#   origin.git  ← feature 分支（作者 clone 推送）
#   branch-module-lab（main）  ← sync.py 拉取并 merge，模块出现在 modules/<lang>/<name>/
#   三语言宿主 ← 从 modules/ 动态加载模块并执行
#
# 只操作 branch-module-lab 这个实验仓库本身（会重置 main 到 demo-base tag），
# 不会动上层 agent 仓库。
set -euo pipefail

LAB="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/bml-demo.XXXXXX")"
ORIGIN="$WORK/origin.git"
AUTHOR="$WORK/author"
LOG="$LAB/demo/logs"
PIDS=()
FAIL=0

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  rm -rf "$WORK" 2>/dev/null || true
}
trap cleanup EXIT

check() {
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then
    echo "  ✔ $name"
  else
    echo "  ✘ $name"
    FAIL=$((FAIL + 1))
  fi
}

contains() { grep -q -- "$2" "$1"; }

# lock 里的 commit 是否已经和作者三个分支的 HEAD 对齐（= 合并 + 构建 + 写 lock 全部完成）
lock_matches_author() {
  python3 - "$LAB" "$AUTHOR" <<'PY'
import json, subprocess, sys

lab, author = sys.argv[1], sys.argv[2]
pairs = [
    ("greet-go", "feat/greet-go"),
    ("greet-js", "feat/greet-js"),
    ("greet-py", "feat/greet-py"),
]
try:
    lock = {m["name"]: m for m in json.load(open(f"{lab}/modules.lock.json"))["modules"]}
except (OSError, ValueError, KeyError):
    sys.exit(1)
for name, branch in pairs:
    head = subprocess.check_output(["git", "-C", author, "rev-parse", branch], text=True).strip()
    if lock.get(name, {}).get("commit") != head:
        sys.exit(1)
PY
}

cd "$LAB"
echo "== 0. 重置实验仓库到 demo 起点"
git checkout -q -B main demo-base
git reset -q --hard demo-base
git clean -fdq
git clean -fdxq modules 2>/dev/null || true
rm -rf .bin demo/logs
mkdir -p .bin "$LOG"

git remote remove origin 2>/dev/null || true
git init -q --bare "$ORIGIN"
git remote add origin "$ORIGIN"
git push -q origin main
git -C "$ORIGIN" symbolic-ref HEAD refs/heads/main
git clone -q "$ORIGIN" "$AUTHOR"
git -C "$AUTHOR" config user.name "bml-author"
git -C "$AUTHOR" config user.email "author@example.com"

echo
echo "== 1. 作者在各自 feature 分支实现单一功能（v1）"
python3 demo/author_modules.py create "$AUTHOR" | tee "$LOG/author-v1.log"

echo
echo "== 2. sync.py once：把 3 个分支 merge 成 main 上的模块"
python3 sync.py once | tee "$LOG/sync-v1.log"

echo
echo "== 3. 三语言宿主动态加载模块并执行"
go build -o .bin/go-host ./go/host
./.bin/go-host --root "$LAB" | tee "$LOG/host-go-v1.log"
node js/host.mjs --root "$LAB" | tee "$LOG/host-js-v1.log"
python3 -u python/host.py --root "$LAB" | tee "$LOG/host-py-v1.log"

echo
echo "== 4. 阶段 1 验收"
check "Go 模块源码已合并" test -f modules/go/greet-go/greet.go
check "JS 模块源码已合并" test -f modules/js/greet-js/index.mjs
check "Python 模块源码已合并" test -f modules/python/greet-py/impl.py
check "Go 宿主加载到 v1" contains "$LOG/host-go-v1.log" "go v1"
check "JS 宿主加载到 v1" contains "$LOG/host-js-v1.log" "js v1"
check "Python 宿主加载到 v1" contains "$LOG/host-py-v1.log" "python v1"
check "lock 记录了 3 个模块" python3 - "$LAB" <<'PY'
import json, sys
mods = json.load(open(f"{sys.argv[1]}/modules.lock.json"))["modules"]
assert len(mods) == 3 and all(m["commit"] for m in mods), mods
PY

echo
echo "== 5. 自动化：启动 sync.py watch，作者把分支升级到 v2"
python3 -u sync.py watch --interval 1 > "$LOG/sync-watch.log" 2>&1 &
PIDS+=("$!")
sleep 1
python3 demo/author_modules.py bump "$AUTHOR" 2.0.0 | tee "$LOG/author-v2.log"

echo "   等待 sync 自动合并 v2（分支 HEAD → merge → build → lock 全部完成）…"
V2_LOCK=0
for _ in $(seq 1 60); do
  if lock_matches_author; then
    V2_LOCK=1
    break
  fi
  sleep 1
done

echo
echo "== 6. 阶段 2 验收：分支更新后，merge 的模块跟着更新"
check "sync watch 自动更新 lock 到 v2" test "$V2_LOCK" = 1
check "Go 模块源码 v2" contains modules/go/greet-go/greet.go "go v2"
check "JS 模块源码 v2" contains modules/js/greet-js/index.mjs "js v2"
check "Python 模块源码 v2" contains modules/python/greet-py/impl.py "python v2"
./.bin/go-host --root "$LAB" | tee "$LOG/host-go-v2.log"
node js/host.mjs --root "$LAB" | tee "$LOG/host-js-v2.log"
python3 -u python/host.py --root "$LAB" | tee "$LOG/host-py-v2.log"
check "Go 宿主输出 v2" contains "$LOG/host-go-v2.log" "go v2"
check "JS 宿主输出 v2" contains "$LOG/host-js-v2.log" "js v2"
check "Python 宿主输出 v2" contains "$LOG/host-py-v2.log" "python v2"

echo
echo "== 7. 全链路自动化：三语言宿主 --watch + 作者升级 v3"
./.bin/go-host --root "$LAB" --watch > "$LOG/host-go-watch.log" 2>&1 &
PIDS+=("$!")
node js/host.mjs --root "$LAB" --watch > "$LOG/host-js-watch.log" 2>&1 &
PIDS+=("$!")
python3 -u python/host.py --root "$LAB" --watch > "$LOG/host-py-watch.log" 2>&1 &
PIDS+=("$!")
sleep 2
python3 demo/author_modules.py bump "$AUTHOR" 3.0.0 | tee "$LOG/author-v3.log"

echo "   等待 sync 合并 v3 …"
V3_LOCK=0
for _ in $(seq 1 60); do
  if lock_matches_author; then
    V3_LOCK=1
    break
  fi
  sleep 1
done
check "sync watch 自动更新 lock 到 v3" test "$V3_LOCK" = 1

echo "   等待三种运行时热更新到 v3 …"
V3_RUNTIME=0
for _ in $(seq 1 40); do
  if grep -q "go v3" "$LOG/host-go-watch.log" 2>/dev/null \
    && grep -q "js v3" "$LOG/host-js-watch.log" 2>/dev/null \
    && grep -q "python v3" "$LOG/host-py-watch.log" 2>/dev/null; then
    V3_RUNTIME=1
    break
  fi
  sleep 1
done
check "Go 插件运行时热更新到 v3" contains "$LOG/host-go-watch.log" "go v3"
check "JS 运行时热更新到 v3" contains "$LOG/host-js-watch.log" "js v3"
check "Python 运行时热更新到 v3" contains "$LOG/host-py-watch.log" "python v3"
check "全链路（分支 push → sync merge → 运行时更新）完成" test "$V3_RUNTIME" = 1

echo
echo "== 8. 最终状态"
git log --oneline --graph -16 | tee "$LOG/git-log.txt"
python3 sync.py status | tee "$LOG/status.txt"
echo
echo "日志目录: $LOG"
if [ "$FAIL" -gt 0 ]; then
  echo "结果: ✘ $FAIL 项失败"
  exit 1
fi
echo "结果: ✔ 全部通过"
