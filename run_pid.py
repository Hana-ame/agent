import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_db import PromptDB
from resolve_prompt import resolve_prompt

db = PromptDB()
pid = int(sys.argv[1]) if len(sys.argv) > 1 else 214
row = db.get(pid)
if not row:
    print(f"#{pid} not found")
    sys.exit(1)

prompt = {"agent": row["agent"] or "Null", "context": row["context"]}

print(f"Input: {json.dumps(prompt, ensure_ascii=False)}")

result = resolve_prompt(prompt, db=db,
    model=row["model"] or "siliconflow-cn/Qwen/Qwen3-8B", timeout=300)

print(f"Output: {result}")

db.done(pid, result, {"source": "manual_run"})
print(f"#{pid} saved")
