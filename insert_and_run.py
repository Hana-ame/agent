import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_db import PromptDB
from resolve_prompt import resolve_prompt

db = PromptDB()

questions = [
    "今天天气怎么样？",
    "你最喜欢什么电影？",
    "如何做番茄炒蛋？",
    "推荐一本好书",
    "用Python写一个斐波那契数列",
    "怎么学英语最有效？",
    "介绍一下量子计算",
]

model_8b = "siliconflow-cn/Qwen/Qwen3-8B"
model_35_4b = "siliconflow-cn/Qwen/Qwen3.5-4B"

pids_8b = []
pids_35 = []

for q in questions:
    pids_8b.append(db.add(q, agent="Null", model=model_8b))
    pids_35.append(db.add(q, agent="Null", model=model_35_4b))

print(f"Inserted {len(questions)} × 2 = {len(pids_8b) + len(pids_35)} entries")

all_pending = db.list_by_status("pending")
print(f"Total pending: {len(all_pending)}")

for row in all_pending:
    pid = row["id"]
    agent = row["agent"] or "Null"
    model = row["model"]
    context = row["context"]

    print(f"\n--- #{pid} ---")
    print(f"  context: {context}")
    print(f"  model:   {model}")

    prompt = {"agent": agent, "context": context}
    t0 = time.time()
    try:
        result = resolve_prompt(prompt, db=db, model=model, timeout=300)
        elapsed = time.time() - t0
        print(f"  output:  {result[:120]}")
        print(f"  time:    {elapsed:.1f}s")
        db.done(pid, result, {"source": "batch_run", "elapsed": round(elapsed, 2)})
    except Exception as e:
        print(f"  ERROR:   {e}")
        db.failed(pid, str(e))

print("\nDone!")
