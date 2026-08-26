# Prompt Mutator & Worker 讲解

## prompt_mutator.py - Context变异守护进程

### 核心功能
使用LLM（大语言模型）自动生成新的context变体，用于prompt优化实验。

### 工作流程
1. **启动**: 运行 `python3 prompt_mutator.py`
2. **循环**: 每30秒执行一次 `process_mutations()`
3. **随机选取**: 从数据库中随机选2条有context的记录
4. **LLM生成**: 调用opencode生成新的context
5. **验证**: 检查JSON格式是否合法
6. **存储**: 新context存入数据库，状态为pending

### 关键函数

```python
def call_llm(prompt_text, timeout=120):
    """调用opencode生成内容"""
    result = opencode_run(prompt_text, timeout=timeout)
    return result.get("output", "")

def generate_mutated_context(base_context, all_ids):
    """使用LLM生成新的context"""
    # 构造prompt，让LLM生成新的context数组
    # 可以包含ID（整数）或文本（字符串）
    response = call_llm(prompt)
    return extract_json_from_response(response)
```

### JSON提取逻辑
`extract_json_from_response()` 从LLM响应中提取JSON：
1. 尝试直接解析
2. 提取 ```json ... ``` 代码块
3. 提取 [ ... ] 或 { ... } 格式
4. 将Python列表语法转换为JSON

---

## prompt_worker.py - Prompt处理守护进程

### 核心功能
持续处理数据库中的pending记录，执行实际的prompt解析任务。

### 工作流程
1. **启动**: 运行 `python3 prompt_worker.py`
2. **循环**: 每2秒执行一次 `process_pending()`
3. **查询**: 获取所有status="pending"的记录
4. **处理**: 调用 `resolve_prompt()` 执行任务
5. **更新**: 处理成功/失败后更新数据库状态

### 关键函数

```python
def process_pending():
    """处理所有pending记录"""
    rows = db.list_by_status("pending")
    for row in rows:
        result = resolve_prompt(pid, db=db, model=model, timeout=300)
        # 处理成功，自动更新状态
```

---

## 两个守护进程的关系

```
┌─────────────────────┐     ┌─────────────────────┐
│   prompt_mutator.py │     │   prompt_worker.py  │
│   (生成context变体)  │     │   (处理pending任务)  │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           ▼                           ▼
        ┌─────────────────────────────────────┐
        │           prompt_db.py              │
        │         (SQLite 数据库)             │
        └─────────────────────────────────────┘
           │                           │
           │  mutator生成新context     │ worker处理pending
           │  状态=pending             │ 状态=done/failed
           ▼                           ▼
        ┌─────────────────────────────────────┐
        │         prompt_lab.db               │
        │   (id, context, status, result)    │
        └─────────────────────────────────────┘
```

### 协作流程
1. **mutator** 生成新context → 存入数据库（status=pending）
2. **worker** 读取pending记录 → 调用resolve_prompt处理
3. **worker** 处理完成 → 更新状态为done/failed
4. 循环往复，持续优化

---

## 运行方式

```bash
# 启动两个守护进程（分别在两个终端）
python3 prompt_mutator.py
python3 prompt_worker.py

# 停止：Ctrl+C 优雅退出
```

## 依赖关系
- `prompt_db.py`: 数据库操作
- `resolve_prompt.py`: 实际的prompt解析逻辑
- `opencode.py`: 调用LLM的接口
