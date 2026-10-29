#!/usr/bin/env bash
# 一次性端到端验证：server + worker.py 的真实拉取链路（不写入 data/）
set -u
cd "$(dirname "$0")/.."
PORT=8412
TMP="$(cd /tmp && pwd)/tp-e2e-$$"
ROOT_DIR="$TMP/root"
mkdir -p "$ROOT_DIR"

TASKPAD_PORT=$PORT TASKPAD_DATA_DIR="$TMP/data" node server.js > "$TMP/server.log" 2>&1 &
SRV=$!
sleep 2

api() { curl -s -X "$1" "http://127.0.0.1:$PORT$2" -H 'Content-Type: application/json' -d "$3"; }

A_ID=$(api POST /api/tasks '{"name":"上游","command":"echo A_PAYLOAD"}' | node -e "let s='';process.stdin.on('data',d=>s+=d).on('end',()=>console.log(JSON.parse(s).task.id))")
B_ID=$(api POST /api/tasks "{\"name\":\"下游\",\"command\":\"python -c \\\"import os;print('RECV='+os.environ.get('__IN_dep','MISSING'))\\\"\"}" | node -e "let s='';process.stdin.on('data',d=>s+=d).on('end',()=>console.log(JSON.parse(s).task.id))")
EDGE=$(api POST /api/edges "{\"from\":\"$A_ID\",\"to\":\"$B_ID\",\"name\":\"成果\",\"kind\":\"data\",\"tests\":[{\"name\":\"接口存活\",\"command\":\"true\"}]}" | node -e "let s='';process.stdin.on('data',d=>s+=d).on('end',()=>console.log(JSON.parse(s).edge.id))")

echo "tasks: $A_ID $B_ID edge: $EDGE"
api POST "/api/tasks/$A_ID/run" '{}' > /dev/null

echo "--- worker 跑第 1 个任务 ---"
python worker.py --server "http://127.0.0.1:$PORT" --worker-id e2e --root "$ROOT_DIR" --once --heartbeat 1 2>&1 | tail -6

api POST "/api/tasks/$B_ID/run" '{}' > /dev/null
echo "--- worker 跑第 2 个任务（应注入上游文本）---"
python worker.py --server "http://127.0.0.1:$PORT" --worker-id e2e --root "$ROOT_DIR" --once --heartbeat 1 2>&1 | tail -8

echo "--- 第 3 个任务：输入门禁必须拦住（input test exit 1）---"
C_ID=$(api POST /api/tasks '{"name":"门禁验证","command":"echo SHOULD_NOT_RUN","inputs":[{"id":"in","name":"dep","tests":[{"name":"输入必过","command":"exit 1"}]}]}' | node -e "let s='';process.stdin.on('data',d=>s+=d).on('end',()=>console.log(JSON.parse(s).task.id))")
api POST /api/edges "{\"from\":\"$B_ID\",\"to\":\"$C_ID\",\"name\":\"下游成果\",\"kind\":\"data\"}" > /dev/null
api POST "/api/tasks/$C_ID/run" '{}' > /dev/null
python worker.py --server "http://127.0.0.1:$PORT" --worker-id e2e --root "$ROOT_DIR" --once --heartbeat 1 2>&1 | tail -6

echo "--- 三次运行结果 ---"
curl -s "http://127.0.0.1:$PORT/api/runs" | node -e "
let s='';process.stdin.on('data',d=>s+=d).on('end',()=>{
  const runs=JSON.parse(s).runs.slice().reverse();
  for(const r of runs) console.log(r.taskName, '->', r.status, '| gate', r.gate? r.gate.ok: 'n/a', '| chars', r.outputChars);
});"
curl -s "http://127.0.0.1:$PORT/api/runs" | node -e "
let s='';process.stdin.on('data',d=>s+=d).on('end',()=>{
  const b=JSON.parse(s).runs.find(r=>r.taskName==='下游');
  const id=b.id; require('http').get('http://127.0.0.1:$PORT/api/runs/'+id,(res)=>{let t='';res.on('data',d=>t+=d).on('end',()=>{
    const run=JSON.parse(t).run; console.log('下游 stdout:', JSON.stringify(run.output.map(o=>o.d).join('').trim()));
  });});
});"

kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
rm -rf "$TMP"
