#!/bin/bash
# ============================================================================
# HN V4 — 纯 curl API 运行脚本
# 用法: bash examples/hn_v4/run_curl.sh [--port 12434]
# ============================================================================

set -e

PORT="${2:-12434}"
BASE="http://127.0.0.1:${PORT}"
SID="hn_v4"

echo "═══════════════════════════════════════════════════════════════"
echo "  HN V4 — curl API Pipeline"
echo "  Server: $BASE"
echo "  Session: $SID"
echo "═══════════════════════════════════════════════════════════════"

# --- 检查服务器是否运行 ---
if ! curl -s "$BASE/api/db/stats" > /dev/null 2>&1; then
  echo "❌ 服务器未启动，请先运行:"
  echo "   python3 -m framework.server_v4 --port $PORT --db /tmp/hn_v4_test.db"
  exit 1
fi

# --- Step 1: 创建顶点 ---
echo ""
echo "▶ Step 1: 创建 7 顶点"

curl -s -X POST "$BASE/api/sessions/$SID/graph/vertices" \
  -H "Content-Type: application/json" \
  -d '{"name":"v_trigger","content":"{\"topic\":\"AI, Systems, Agents\",\"limit\":6}","attributes":["start"],"state":"data ready"}' > /dev/null

for v in v_raw_stories v_filtered_stories v_discussions v_sys_env v_final_report; do
  if [ "$v" = "v_final_report" ]; then
    attrs='["end","plain text"]'
  else
    attrs='["json"]'
  fi
  curl -s -X POST "$BASE/api/sessions/$SID/graph/vertices" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$v\",\"content\":\"\",\"attributes\":$attrs,\"state\":\"todo\"}" > /dev/null
done

curl -s -X POST "$BASE/api/sessions/$SID/graph/vertices" \
  -H "Content-Type: application/json" \
  -d '{"name":"v_context_bundle","content":"{}","attributes":["json"],"state":"todo"}' > /dev/null

echo "  ✅ 7 vertices created"

# --- Step 2: 创建边 ---
echo ""
echo "▶ Step 2: 创建 8 边"

# Branch A: HN Feed
curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_fetch_top","type":"code","input_vertex":"v_trigger","output_vertex":"v_raw_stories","script":"examples/hn_v4/hn_transforms.py:fetch_hn_top_stories","settings":{"limit":6,"timeout":20.0}}' > /dev/null

curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_filter","type":"code","input_vertex":"v_raw_stories","output_vertex":"v_filtered_stories","script":"examples/hn_v4/hn_transforms.py:filter_stories","settings":{"max_selected":3,"timeout":20.0}}' > /dev/null

curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_comments","type":"code","input_vertex":"v_filtered_stories","output_vertex":"v_discussions","script":"examples/hn_v4/hn_transforms.py:fetch_comments_and_context","settings":{"timeout":20.0}}' > /dev/null

curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_merge_hn","type":"code","input_vertex":"v_discussions","output_vertex":"v_context_bundle","script":"examples/hn_v4/hn_transforms.py:package_hn_data","settings":{"merge_strategy":"json_merge"}}' > /dev/null

# Branch B: Host Telemetry
curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_sys_probe","type":"code","input_vertex":"v_trigger","output_vertex":"v_sys_env","script":"examples/hn_v4/hn_transforms.py:probe_system_environment"}' > /dev/null

curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_merge_sys","type":"code","input_vertex":"v_sys_env","output_vertex":"v_context_bundle","script":"examples/hn_v4/hn_transforms.py:package_sys_data","settings":{"merge_strategy":"json_merge"}}' > /dev/null

# Final Report
curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_report","type":"code","input_vertex":"v_context_bundle","output_vertex":"v_final_report","script":"examples/hn_v4/hn_transforms.py:generate_markdown_report"}' > /dev/null

# Reflexive Self-Healing
curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_recover_fetch","type":"reflexive","input_vertex":"v_raw_stories","output_vertex":"v_raw_stories","script":"examples/hn_v4/hn_transforms.py:recovery_fetch_fallback","trigger_state":"reject","target_state":"data ready","max_retries":2}' > /dev/null

echo "  ✅ 8 edges created"

# --- Step 3: 验证 DAG ---
echo ""
echo "▶ Step 3: 验证 DAG"
VALIDATE=$(curl -s -X POST "$BASE/api/sessions/$SID/graph/validate")
echo "$VALIDATE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Valid: {d[\"valid\"]}')
print(f'  Vertices: {d[\"vertex_count\"]}')
print(f'  Edges: {d[\"edge_count\"]}')
print(f'  Tiers: {json.dumps(d[\"tiers\"], indent=4)}')
"

# --- Step 4: 执行 ---
echo ""
echo "▶ Step 4: 执行流水线"
RESULT=$(curl -s -X POST "$BASE/api/sessions/$SID/run" \
  -H "Content-Type: application/json" \
  -d '{"timeout":120,"max_concurrency":4}')

echo "$RESULT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Success: {d[\"success\"]}')
print(f'  Time: {d[\"execution_time\"]:.2f}s')
print(f'  Edges: {len(d[\"completed_edges\"])} completed')
print(f'  Errors: {len(d[\"errors\"])}')
for eid, er in d['edge_results'].items():
    status = '✅' if er['success'] else '❌'
    print(f'    {status} {eid}')
print(f'  States: {json.dumps(d[\"vertex_states\"], indent=4)}')
"

# --- Step 5: 获取报告 ---
echo ""
echo "▶ Step 5: 获取报告"
REPORT=$(curl -s "$BASE/api/db/sessions/$SID/vertices?v_name=v_final_report")
REPORT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "$REPORT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
vertices = data.get('vertices', data if isinstance(data, list) else [])
for v in vertices:
    if v.get('name') == 'v_final_report':
        print(v['content'])
" > "$REPORT_DIR/curl_report.md"

echo "  ✅ 报告已保存到 $REPORT_DIR/curl_report.md"

# --- 完成 ---
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ 完成 — 报告: $REPORT_DIR/curl_report.md"
echo "═══════════════════════════════════════════════════════════════"
