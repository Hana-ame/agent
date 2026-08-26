# Run Complete: Entries 214, 215, 216

## Method

```python
row = db.get(pid)
prompt = {"agent": row["agent"] or "Null", "context": row["context"]}
result = resolve_prompt(prompt, db=db, model=row["model"] or "opencode/mimo-v2.5-free", timeout=300)
db.done(pid, result, {"source": "manual_run"})
```

Key: `resolve_prompt.py` now parses string context as JSON before dispatching, so `"[212, 213]"` → `[212, 213]` → resolves references.

## Results

| ID | Context | Output |
|----|---------|--------|
| 214 | `[212, 213]` → 212("蓝色是我最喜欢的颜色。") + 213("面条是我最喜欢的食物。") | 蓝色和面条，很棒的搭配。有什么我可以帮你的吗？ |
| 215 | `test` | Hello! System is working. How can I help you today? |
| 216 | `1+1=?` | 2 |

## Files

- `run_pid.py` — reusable script to run any entry by ID
- `resolve_prompt.py` — fix at lines 160-167: JSON string context auto-parse
