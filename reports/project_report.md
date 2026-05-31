# simpleAI 项目报告

**日期**: 2026-05-31
**分支**: main-new
**测试结果**: 17/17 通过 ✅
**代码检查**: ruff 0 警告，pyright 0 错误 ✅

---

## 一、项目概述

simpleAI 是一个免费模型调用追踪系统，核心功能：
- 从 OpenCode Zen API 和 NVIDIA NIM API 动态发现免费模型
- 记录每个模型的调用统计（成功/失败/质量好评/差评）
- 通过 `opencode` CLI 封装调用，支持 agent/model 选择

---

## 二、文件清单

| 文件 | 类型 | 行数 | 说明 |
|------|------|------|------|
| `db.py` | 核心 | 11 | SQLite 统一入口，WAL 模式，外键开启 |
| `model_tracker.py` | 核心 | 195 | 模型发现 + 调用统计 CRUD，含类型注解 |
| `opencode.py` | 核心 | 38 | opencode CLI 封装（run / models） |
| `test.py` | 测试 | 82 | opencode.py 单元测试（8 个，mock） |
| `test_model_tracker.py` | 测试 | 188 | model_tracker.py 单元测试（8 个，真 DB） |
| `test_flow.py` | 集成 | 34 | 真实调用 opencode 运行模型（1 个） |
| `_demo_model_tracker.py` | 脚本 | 47 | model_tracker 功能演示 |
| `.gitignore` | 配置 | 6 | 排除 __pycache__ / .opencode / *.db |
| `reports/project_report.md` | 文档 | 本文件 | 项目报告 |

---

## 三、模块详解

### db.py — 数据库入口
```python
db.get_conn()  # 返回 sqlite3 连接，WAL + 外键
```
- 数据库文件: `simpleai.db`（同目录）

### model_tracker.py — 模型追踪

**表结构:**
- `models`: model(PK), provider, discovered_at, last_seen
- `usage`: model(PK/FK), calls, successes, failures, good, bad

**API:**
| 函数 | 返回类型 | 说明 |
|------|----------|------|
| `sync_models()` | `list[str]` | 从 API 拉取模型列表，写入 DB |
| `list_free_models()` | `list[str]` | 返回所有模型，自动补全 usage(0,0,0,0,0) |
| `record_call(model, success, good, bad)` | `None` | 记录一次调用，自动累加 |
| `get_stats(model: str)` | `dict \| None` | 查询指定模型统计 |
| `get_stats()` | `list[dict]` | 查询全部模型统计 |

**数据源:**
- OpenCode Zen API: `https://opencode.ai/zen/v1/models`
- NVIDIA NIM API: `https://integrate.api.nvidia.com/v1/models`
- SiliconFlow: 3 个硬编码模型（Qwen3-8B, GLM-Z1-9B, GLM-4-9B）

### opencode.py — CLI 封装

| 函数 | 说明 |
|------|------|
| `run(prompt, agent, model, timeout)` | 调用 `opencode run`，返回 `{output, json, success}` |
| `models(filter_free=True)` | 调用 `opencode models`，filter_free 时只返回免费模型 |

---

## 四、测试结果

### test.py — opencode.py 单元测试 (8/8 ✅)

| 测试 | 内容 | 结果 |
|------|------|------|
| test_json_output | JSON 输出解析 | ✅ |
| test_non_json_output | 非 JSON 返回原始文本 | ✅ |
| test_failure | 命令失败 success=False | ✅ |
| test_command_no_agent_no_model | 无参命令构造 | ✅ |
| test_command_with_agent | 带 --agent 参数 | ✅ |
| test_command_with_model | 带 --model 参数 | ✅ |
| test_command_with_both | agent+model 同时传 | ✅ |
| test_timeout_passthrough | timeout 透传 | ✅ |

### test_model_tracker.py — model_tracker 单元测试 (8/8 ✅)

| 测试 | 内容 | 结果 |
|------|------|------|
| test_list_free_models_adds_usage_record | 自动补全 usage 记录 | ✅ |
| test_list_free_models_does_not_overwrite_existing | 已有 usage 不覆盖 | ✅ |
| test_record_call_updates_usage | 累加调用统计 | ✅ |
| test_record_call_good_bad | good/bad 独立于 success | ✅ |
| test_get_stats_specific_model | 查询指定模型 | ✅ |
| test_get_stats_nonexistent | 不存在模型返回 None | ✅ |
| test_get_stats_all | 查询全部模型 | ✅ |
| test_record_call_auto_insert | 自动插入新模型 | ✅ |

### test_flow.py — 集成测试 (1/1 ✅)

| 测试 | 内容 | 结果 |
|------|------|------|
| test_real_call_qwen3_success | 真实调用 Qwen3-8B，记录并查询统计 | ✅ |

---

## 五、本次会话变更

1. **类型注解**: `model_tracker.py` 全函数添加类型注解 + `@overload`，pyright 零错误
2. **ruff 修复**: 清理 18 个警告（未使用导入、无占位符 f-string、多导入同行）
3. **DB_PATH 修复**: `_demo_model_tracker.py` 中 `fm.DB_PATH` → `db.DB_PATH`
4. **opencode 配置**: 启用 LSP (`lsp: true`)，所有权限默认允许 (`permission: "allow"`)
5. **项目报告**: 新增 `reports/project_report.md`

---

## 六、已知问题 & 待办

1. `test_flow.py` 为集成测试，需真实网络 + opencode CLI
2. `_demo_model_tracker.py` 使用临时 DB `/tmp/test_demo_real.db`，仅为演示
3. `.last_update` 和 `.last_upload_url` 为运行时产物，未在 .gitignore 中排除
