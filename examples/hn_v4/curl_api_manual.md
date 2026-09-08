# HN V4 — curl API 运行手册

> 通过 HTTP API 从零搭建并执行 Hacker News V4 多分支流水线，无需 Python 脚本。

## 前置条件

```bash
# 启动 V4 服务器（端口 12434）
python3 -m framework.server_v4 --port 12434 --db /tmp/hn_v4_test.db
```

## 流水线拓扑

```
v_trigger ─┬─ e_fetch_top ──▶ v_raw_stories ── e_filter ──▶ v_filtered ── e_comments ──▶ v_discussions ── e_merge_hn ─┐
           │                                                                                        (JSON_MERGE)           │
           └─ e_sys_probe ──▶ v_sys_env ────────────────── e_merge_sys ────────────────────────────────────────────▶ v_context_bundle
                                                                                                                                    │
                                                                                                                            e_report
                                                                                                                                    ▼
                                                                                                                              v_final_report
```

- **Branch A (HN Feed)**: 抓取 → 筛选 → 评论 → 打包
- **Branch B (Host Telemetry)**: 系统探测 → 打包
- **Convergence**: `v_context_bundle` 双分支 JSON_MERGE 汇合
- **Fault Tolerance**: `e_recover_fetch` reflexive 自修复（reject → data ready）

## 运行步骤

### Step 1: 创建顶点

```bash
BASE="http://127.0.0.1:12434"
SID="hn_v4"

# 入口顶点（state: data ready）
curl -s -X POST "$BASE/api/sessions/$SID/graph/vertices" \
  -H "Content-Type: application/json" \
  -d '{"name":"v_trigger","content":"{\"topic\":\"AI, Systems, Agents\",\"limit\":6}","attributes":["start"],"state":"data ready"}'

# 中间顶点（state: todo, attributes: json）
for v in v_raw_stories v_filtered_stories v_discussions v_sys_env v_final_report; do
  if [ "$v" = "v_final_report" ]; then
    attrs='["end","plain text"]'
  else
    attrs='["json"]'
  fi
  curl -s -X POST "$BASE/api/sessions/$SID/graph/vertices" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$v\",\"content\":\"\",\"attributes\":$attrs,\"state\":\"todo\"}"
done

# 汇合顶点（空 JSON 对象）
curl -s -X POST "$BASE/api/sessions/$SID/graph/vertices" \
  -H "Content-Type: application/json" \
  -d '{"name":"v_context_bundle","content":"{}","attributes":["json"],"state":"todo"}'
```

### Step 2: 创建边

> **注意**: API 字段是 `id` 和 `type`，不是 `edge_id`/`edge_type`。

```bash
# --- Branch A: HN Feed ---

# e_fetch_top: v_trigger → v_raw_stories
curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_fetch_top","type":"code","input_vertex":"v_trigger","output_vertex":"v_raw_stories","script":"examples/hn_v4/hn_transforms.py:fetch_hn_top_stories","settings":{"limit":6,"timeout":20.0}}'

# e_filter: v_raw_stories → v_filtered_stories
curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_filter","type":"code","input_vertex":"v_raw_stories","output_vertex":"v_filtered_stories","script":"examples/hn_v4/hn_transforms.py:filter_stories","settings":{"max_selected":3,"timeout":20.0}}'

# e_comments: v_filtered_stories → v_discussions
curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_comments","type":"code","input_vertex":"v_filtered_stories","output_vertex":"v_discussions","script":"examples/hn_v4/hn_transforms.py:fetch_comments_and_context","settings":{"timeout":20.0}}'

# e_merge_hn: v_discussions → v_context_bundle (JSON_MERGE)
curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_merge_hn","type":"code","input_vertex":"v_discussions","output_vertex":"v_context_bundle","script":"examples/hn_v4/hn_transforms.py:package_hn_data","settings":{"merge_strategy":"json_merge"}}'

# --- Branch B: Host Telemetry ---

# e_sys_probe: v_trigger → v_sys_env
curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_sys_probe","type":"code","input_vertex":"v_trigger","output_vertex":"v_sys_env","script":"examples/hn_v4/hn_transforms.py:probe_system_environment"}'

# e_merge_sys: v_sys_env → v_context_bundle (JSON_MERGE)
curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_merge_sys","type":"code","input_vertex":"v_sys_env","output_vertex":"v_context_bundle","script":"examples/hn_v4/hn_transforms.py:package_sys_data","settings":{"merge_strategy":"json_merge"}}'

# --- Final Report ---

# e_report: v_context_bundle → v_final_report
curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_report","type":"code","input_vertex":"v_context_bundle","output_vertex":"v_final_report","script":"examples/hn_v4/hn_transforms.py:generate_markdown_report"}'

# --- Reflexive Self-Healing ---

# e_recover_fetch: v_raw_stories → v_raw_stories (self-loop, reject → data ready)
curl -s -X POST "$BASE/api/sessions/$SID/graph/edges" \
  -H "Content-Type: application/json" \
  -d '{"id":"e_recover_fetch","type":"reflexive","input_vertex":"v_raw_stories","output_vertex":"v_raw_stories","script":"examples/hn_v4/hn_transforms.py:recovery_fetch_fallback","trigger_state":"reject","target_state":"data ready","max_retries":2}'
```

### Step 3: 验证 DAG

```bash
curl -s -X POST "$BASE/api/sessions/$SID/graph/validate" | python3 -m json.tool
```

期望输出：
```json
{
  "valid": true,
  "tiers": {
    "e_fetch_top": 0,
    "e_sys_probe": 0,
    "e_recover_fetch": -1,
    "e_filter": 1,
    "e_comments": 2,
    "e_merge_hn": 3,
    "e_merge_sys": 1,
    "e_report": 4
  },
  "vertex_count": 7,
  "edge_count": 8
}
```

### Step 4: 执行

```bash
curl -s -X POST "$BASE/api/sessions/$SID/run" \
  -H "Content-Type: application/json" \
  -d '{"timeout":120,"max_concurrency":4}' | python3 -m json.tool
```

期望输出：
```json
{
  "session_id": "hn_v4",
  "success": true,
  "execution_time": 3.28,
  "completed_edges": ["e_sys_probe","e_merge_sys","e_fetch_top","e_filter","e_comments","e_merge_hn","e_report"],
  "vertex_states": {
    "v_trigger": "data ready",
    "v_raw_stories": "data ready",
    "v_filtered_stories": "data ready",
    "v_discussions": "data ready",
    "v_sys_env": "data ready",
    "v_final_report": "data ready",
    "v_context_bundle": "data ready"
  },
  "errors": []
}
```

### Step 5: 获取报告

```bash
curl -s "$BASE/api/db/sessions/$SID/vertices?v_name=v_final_report" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
for v in data.get('vertices', data if isinstance(data, list) else []):
    if v.get('name') == 'v_final_report':
        print(v['content'])
" > report.md
```

## 一键脚本

```bash
# 复制到项目根目录后运行
bash run_hn_v4_curl.sh
```

## API 参考

| 操作 | 方法 | 路径 |
|---|---|---|
| 创建顶点 | POST | `/api/sessions/{sid}/graph/vertices` |
| 创建边 | POST | `/api/sessions/{sid}/graph/edges` |
| 验证 DAG | POST | `/api/sessions/{sid}/graph/validate` |
| 执行 | POST | `/api/sessions/{sid}/run` |
| 查询顶点 | GET | `/api/db/sessions/{sid}/vertices?v_name={name}` |
| 查询边 | GET | `/api/db/sessions/{sid}/edges` |
| 查询边指标 | GET | `/api/db/sessions/{sid}/metrics` |
| 查询快照 | GET | `/api/db/sessions/{sid}/snapshots` |
| Dashboard | GET | `/dashboard` |
| OpenAI 兼容 | POST | `/v1/chat/completions` |

## 关键注意事项

1. **边字段名**: API 用 `id` / `type`，不是 `edge_id` / `edge_type`
2. **状态值**: 用 `"data ready"` 不是 `"data_ready"`
3. **script 路径**: 相对于服务器 CWD，不是相对于 config.json
4. **JSON_MERGE**: `settings.merge_strategy = "json_merge"` 让 fan-in 顶点合并多分支数据
5. **reflexive 边**: `input_vertex == output_vertex` + `trigger_state` + `target_state` 实现自修复
