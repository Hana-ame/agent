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

## 文件结构
- `html/index.html` — 前端页面（含可折叠 API 输入框，默认地址 https://wsl-8000.moonchan.xyz）
- `html/ping.html` — 连接测试页，调用 `/ping` 端点
- `html/readme.md` — 上传记录

## 注意事项
- `ping.html` 报 `failed to fetch`：确保 FastAPI 正在运行，且 `app.py` 需包含 CORS 中间件（`allow_origins=["*"]`）
- 前端页面请求 API 时，后端必须先启动
- **数据库文件 `state.db` 不得随意删除，除非用户明确指令**

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
| 6 | pagination newer | `curl -x "" "http://localhost:8000/api/prompts?id_gt=5&limit=3"` | 按 id > 查询 |
- `html/prompt.html` — Prompt 管理页

