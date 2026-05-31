# 测试报告: simpleAI 模块

**日期**: 2026-05-31
**运行命令**: `python -m pytest test.py test_model_tracker.py -v -s`
**结果**: 19/19 通过 ✅

---

## 一、opencode.py 测试 (test.py)

| 测试名 | 测试内容 | 结果 |
|--------|---------|------|
| test_json_output | 解析 JSON 输出，返回 `{"output": ..., "json": True, "success": True}` | ✅ |
| test_non_json_output | 非 JSON 输出返回原始文本，`json=False` | ✅ |
| test_failure | 命令失败时 `success=False` | ✅ |
| test_command_no_agent_no_model | 构造 `["opencode", "run", "hello"]` | ✅ |
| test_command_with_agent | 构造 `["opencode", "--agent", "Null", "run", "hello"]` | ✅ |
| test_command_with_model | 构造 `["opencode", "--model", "gemma-4-31b-it", "run", "hello"]` | ✅ |
| test_command_with_both | 同时传 agent+model 时参数顺序正确 | ✅ |
| test_timeout_passthrough | timeout=999 透传到 subprocess | ✅ |

---

## 二、model_tracker.py 测试 (test_model_tracker.py)

使用 NVIDIA 图片模型（flux 系列）做测试数据，opencode 永远不会调用这些模型，测试完自动清理。

| 测试名 | 测试内容 | 结果 |
|--------|---------|------|
| test_list_free_models_empty | 新模型不在 DB 中 → 查询返回空 | ✅ |
| test_list_free_models_with_data | 插入 2 个模型后 `list_free_models` 能正确列出 | ✅ |
| test_get_stats_all_empty | 模型无 usage 记录 → calls=0, successes=0, failures=0 | ✅ |
| test_get_stats_specific | 查询指定模型统计，返回 calls=5, successes=4, failures=1 | ✅ |
| test_get_stats_nonexistent | 查询不存在的模型 → None | ✅ |
| test_record_call_success | `record_call(success=True)` → calls+1, successes+1 | ✅ |
| test_record_call_failure | `record_call(success=False)` → calls+1, failures+1 | ✅ |
| test_record_call_accumulate | 3 次调用后累加正确: calls=3, successes=2, failures=1 | ✅ |
| test_record_call_auto_insert | 自动 INSERT 不存在的模型到 DB | ✅ |
| test_get_stats_all_with_usage | `get_stats()` 返回全部模型统计，包含有/无 usage 的 | ✅ |
| test_list_free_models_after_sync | `sync_models` 结果被 `list_free_models` 包含 | ✅ |

---

## 三、模块功能说明

### opencode.py
- `run(prompt, agent, model, timeout)` — 调用 `opencode run` CLI，返回 JSON 或原始文本
- `models(filter_free)` — 调用 `opencode models`，filter_free=True 时只返回 opencode/* + nvidia/* + 3个特定 siliconflow 模型

### model_tracker.py
- `list_free_models()` — 查询 DB 中所有模型，确保每条都有 usage 记录 (0,0,0)
- `get_stats(model)` — 查询指定/全部模型的调用统计
- `record_call(model, success)` — 记录一次调用成功/失败，自动累加

### db.py
- 统一数据库入口 `simpleai.db`，WAL 模式，外键开启

---

## 四、关键设计

1. **统一 DB**: 所有模块共用 `simpleai.db`，通过 `db.py` 获取连接
2. **测试隔离**: 每个测试前后自动清理 NVIDIA flux 模型数据，不影响真实数据
3. **真实 DB**: 不使用临时 DB，测试直接操作 `simpleai.db`
