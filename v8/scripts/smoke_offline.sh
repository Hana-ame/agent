#!/usr/bin/env bash
# 无网关自检：所有不依赖真实模型的路径跑一遍。
#   bash scripts/smoke_offline.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== pytest（离线，跳过 live） =="
python3 -m pytest tests/ -q -m "not slow" --ignore=tests/test_live_pi.py

echo "== README 计数一致性 =="
python3 scripts/check_docs.py

echo "== 离线 demo =="
python3 -m dag8 demo >/dev/null

echo "== doctor（只查本地） =="
python3 -m dag8 doctor --skip-gateway

echo "== 模板：逐个生成 + dry-run =="
for t in $(python3 -m dag8 new --list | awk '{print $1}'); do
  out="/tmp/dag8_smoke_${t}.json"
  python3 -m dag8 new "$t" --out "$out" >/dev/null
  python3 -m dag8 run "$out" --db /tmp/dag8_smoke.sqlite3 --dry-run >/dev/null
  echo "  ok $t"
done

echo "== scaffold --compose + lint + dry-run =="
rm -rf /tmp/dag8_smoke_arch
python3 -m dag8 scaffold . --mode dir --compose --max-modules 4 \
  --out /tmp/dag8_smoke_arch/project.json >/dev/null
python3 -m dag8 lint /tmp/dag8_smoke_arch/project.json
python3 -m dag8 --db /tmp/dag8_smoke.sqlite3 run /tmp/dag8_smoke_arch/project.json --dry-run >/dev/null

echo "== convert v7 → v8 + lint =="
python3 -m dag8 convert configs/graphs/v7_tighten.json \
  --out /tmp/dag8_smoke_v7.json >/dev/null
python3 -m dag8 lint /tmp/dag8_smoke_v7.json

echo "OK：离线链路全部通过"
