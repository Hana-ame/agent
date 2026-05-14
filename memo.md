# Memo

## 项目概况
- FastAPI 项目，监听端口 8000
- 公网访问地址：`http://wsl-8000.moonchan.xyz`

## 本地测试
- 使用 `curl -x ""` 绕过代理，例如：`curl -x "" http://localhost:8000/api/...`

## 规则
- `html/` 文件夹中的每次修改，都要上传到 `upload.moonchan.xyz`
- `html/readme.md` 记录所有上传链接
- 所有新建的 `.html` 文件都必须包含可折叠的 API Endpoint 输入框
- **`html/` 内文件每次修改后，必须重新 upload 并更新 `readme.md`**

## Node 设计
- `accept_tags` 为 AND 逻辑：必须**所有** tag 都有未处理数据，节点才执行
- `{tag_name}` 在 prompt 中作为变量，会被替换为对应 tag 的文本值
- `output_tag` 为单个输出 tag

## 文件结构
- `html/index.html` — 前端页面（含可折叠 API 输入框，默认地址 https://wsl-8000.moonchan.xyz）
- `html/ping.html` — 连接测试页，调用 `/ping` 端点
- `html/readme.md` — 上传记录

## 注意事项
- `ping.html` 报 `failed to fetch`：确保 FastAPI 正在运行，且 `app.py` 需包含 CORS 中间件（`allow_origins=["*"]`）
- 前端页面请求 API 时，后端必须先启动
- **数据库文件 `state.db` 不得随意删除，除非用户明确指令**
- **每次修改数据库表结构后，必须同步更新 `create_db.py`**

## 工作纪律（必须遵守）
- 修改 `readme.md` 时只追加新行，**不得删除或覆盖已有条目**
- `html/` 内文件每次修改后必须重新 upload 并更新 `readme.md`
- upload 前先确认本地文件是最新版
- 不确定用户意图时先问，不要猜
- 涉及删除/覆盖已有数据或文件的修改，先确认再动手

## Git 工作流
- 提交前必须先执行所有已记录的测试，通过后才能 commit

## 测试清单

| 编号 | 名称 | 命令 | 预期结果 |
|------|------|------|----------|
| 0 | ping/pong | `curl -x "" http://localhost:8000/ping` | 返回 `pong` |
| 1 | POST prompt | `curl -x "" -X POST http://localhost:8000/api/prompts -H "Content-Type: application/json" -d '{"tag":"t","prompt":"test"}'` | 返回 `{"status":"ok"}` |
| 2 | GET prompts | `curl -x "" http://localhost:8000/api/prompts` | 返回 JSON 列表 |
| 4 | pagination init | `curl -x "" "http://localhost:8000/api/prompts?limit=3"` | `has_older`/`has_newer` 正确 |
| 5 | pagination older | `curl -x "" "http://localhost:8000/api/prompts?id_lt=8&limit=3"` | 按 id < 查询 |
| 7 | Node CRUD create | `curl -x "" -X POST http://localhost:8000/api/nodes -H "Content-Type: application/json" -d '{"name":"n","accept_tags":"t1,t2","output_tag":"out","model":"m","prompt":"{t1}","interval":5}'` | `{"status":"ok"}` |
| 8 | Node CRUD list | `curl -x "" http://localhost:8000/api/nodes` | JSON 列表含 model |
| 9 | Node CRUD update | `curl -x "" -X PUT http://localhost:8000/api/nodes/1 -H "Content-Type: application/json" -d '{"name":"n2","accept_tags":"t1","output_tag":"o","model":"m2","prompt":"","interval":3}'` | `{"status":"ok"}` |
| 10 | Node CRUD delete | `curl -x "" -X DELETE http://localhost:8000/api/nodes/1` | `{"status":"ok"}` |
| 11 | call_opencode | `python3 -c "from app import call_opencode; r=call_opencode('hi','...'); print(r['success'])"` | `True` + usage |
| 12 | Node Exec list | `curl -x "" "http://localhost:8000/api/execs?limit=10"` | JSON `items` 列表 |
| 13 | Node Exec filter | `curl -x "" "http://localhost:8000/api/execs?node_name=responder"` | 仅返回该 node 记录 |
| 14 | poll round-trip | 创建 node(accept_tags="test_in", output_tag="test_out", interval=5) → POST prompt(tag="test_in") → 5s 后 GET execs 有新记录 | 自动执行并生成 output prompt |
- `html/prompt.html` — Prompt 管理页
- `html/node.html` — Node 管理页（可接受多tag、输出tag、{[tag]}替换、间隔）
- `create_db.py` — 独立数据库初始化脚本（`python create_db.py --seed`）

