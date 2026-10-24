#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
V8=$(cd "$DIR/../.." && pwd)
if [ $# -lt 1 ]; then
  echo "用法: $0 report.md [额外 dag8 run 参数，如 --json]"
  exit 2
fi
REPORT=$1
shift
cd "$V8"
python3 -m dag8 run configs/report_check/dag.json --seed "$(cat "$REPORT")" "$@"
