# resolve_prompt — DAG 节点调度器

**日期**: 2026-06-11
**版本**: v2 (DAG 调度器架构)
**唯一可用模块**: `resolve_prompt.py`（`resolve_with_db.py` 已废弃）

---

## 一、项目概述

`resolve_prompt.py` 是一个**递归 Prompt 解析器 / DAG 节点调度器**，核心能力：

- 将 Prompt 定义为由 `int`（数据库 ID 引用）、`str`、`dict`（Agent 任务块）、`list` 组成的嵌套结构
- 自动解析依赖树：引用 `done` 节点时只提取 `response`（纯净结果），引用 `pending` 节点时自动补跑
- 最终拼接解析后的上下文，调用 `opencode` API 获取结果

---

## 二、架构设计

```
resolve_prompt({"agent": "Null", "context": [pid1, "text", {...}]})
       │
       ▼
  _resolve_element(context)     ← 统一递归引擎
       │
       ├─ int  → _resolve_int(pid)
       │           ├─ done    → 只返回 response（纯净提取）
       │           └─ pending → 递归跑完再返回
       │
       ├─ str  → 原样保留
       │
       ├─ list → 每个元素递归 _resolve_element，\n\n 拼接
       │
       └─ dict → 递归解析 context → API → 返回结果
       │
       ▼
  opencode_run(resolved_text, agent=agent)
       │
       ▼
  return output
```

---

## 三、文件清单

| 文件 | 说明 |
|------|------|
| `resolve_prompt.py` | **唯一可用模块**。DAG 调度器，入口收窄为只接受 dict/JSON str |
| `prompt_db.py` | SQLite 持久层，prompts 表（id/context/agent/model/response/log/status/score/elo） |
| `opencode.py` | opencode CLI 封装（`run` / `models`） |
| `test_resolve_prompt.py` | 详细调试测试，三段式输出：【输入】→【API 调用】→【最终返回值】 |

---

## 四、API 参考

### `resolve_prompt(prompt_input, *, db, model, timeout)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `prompt_input` | `str \| dict` | **仅接受 JSON 字符串或 dict**。裸 int/str 报错。 |
| `db` | `PromptDB` | 数据库实例 |
| `model` | `str` | 默认模型名 |
| `timeout` | `int` | API 超时秒数（默认 600） |

**返回值**: `str` — API 输出文本

### `_resolve_int(pid, db, model, timeout)`

解析数据库 ID 引用。`done` 状态只返回 `response`；`pending` 状态先递归执行再返回结果。

### `_resolve_element(element, db, model, timeout)`

统一递归引擎。按 `int → _resolve_int` / `str → 原样` / `list → 逐个递归后拼接` / `dict → 解析 context 后调 API` 分发。

---

## 五、测试结果

### 测试场景（6/6 ✅ 真实 API）

| # | 场景 | 输入 | 验证要点 |
|---|------|------|----------|
| 1 | dict + int（引用已有 resp） | `{"agent":"Null","context": pid1}` | resp 被纯净提取作为 API 输入 |
| 2 | list 混合 str + int | `{"context":["水果",pid1,"蔬菜",pid2]}` | 正确拼接后送 API |
| 3 | 嵌套 dict（pending 自动补跑） | `{"context":[pid3(pending),"请总结"]}` | pending 节点先执行再拼接 |
| 4 | 纯文本 context | `{"context":"1+1等于几？"}` | 直送 API |
| 5 | 嵌套 dict + int 引用 | `{"context":["开头",{"context":"嵌套问题"}]}` | 嵌套递归正确 |
| 6 | 嵌套 dict 引用 int | `{"context":[{"context":pid1}]}` | dict 包 int 引用正确 |

---

## 六、核心设计决策

1. **入口收紧** — 只接受 `dict` 或 JSON 字符串，裸 int/str 报错，强制结构化
2. **纯净提取** — `_resolve_int` 对 done 条目只返回 `response`，不掺杂 `context`
3. **Pending 自动补跑** — DAG 特性：依赖未就绪时自动递归先跑
4. **统一 `_resolve_element`** — int/str/list/dict 统一分发，消除重复逻辑
5. **废弃 class 版** — `resolve_with_db.py` (`PromptResolver`) 不再维护

---

## 七、已知问题

- API 调用超时（`subprocess.TimeoutExpired`）未在 `opencode.run` 中捕获，会直接崩溃
- `prompt_db.py` 中 `prompts` 表含 `score` / `elo` 字段，当前调度器未使用

---

*完整测试日志: 见 Board 666 no:193200*
