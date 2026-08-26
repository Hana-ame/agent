"""1→读DB 2→插pending 3→读DB 4→执行+输出 5→读DB"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_db import PromptDB
from resolve_prompt import resolve_prompt

db = PromptDB()
MODEL = "siliconflow-cn/Qwen/Qwen3-8B"

def dump(msg=""):
    if msg: print(f"\n{'='*60}\n{msg}\n{'='*60}")
    print(f"{'ID':>4} {'status':<8} {'context':<50} {'response':<50}")
    print("-" * 120)
    for r in db.list_all():
        ctx = r["context"][:48] + ".." if len(r["context"]) > 48 else r["context"]
        resp = r["response"][:48] + ".." if len(r["response"]) > 48 else r["response"]
        print(f"{r['id']:>4} {r['status']:<8} {ctx:<50} {resp:<50}")
    print(f"\n总计: {len(db.list_all())} 条")

# ── 0. 清空 DB ──
with db._conn() as conn:
    conn.execute("DELETE FROM prompts")
    conn.commit()
print("⑩ 清空 DB 完成\n")

# ── 1. 读DB ──
dump("① 读取整个 DB（初始状态）")

# ── 2. 插pending ──
pid_a = db.add("你最喜欢什么颜色？", agent="Qwen3-8B")
db.done(pid_a, "蓝色是我最喜欢的颜色。")
pid_b = db.add("你最喜欢什么食物？", agent="Qwen3-8B")
db.done(pid_b, "面条是我最喜欢的食物。")

# 3种pending: 纯文本 / list引用 / 嵌套
pid_p1 = db.add("1+1等于几？请直接回答数字", agent="Qwen3-8B")
pid_p2 = db.add([pid_a, pid_b], agent="Qwen3-8B")
pid_p3 = db.add("中国首都是哪里？请直接回答城市名", agent="Qwen3-8B")
print(f"\n>>> 插入了 3 条 pending: #{pid_p1} #{pid_p2} #{pid_p3}")

# ── 3. 读DB ──
dump("③ 插入后读取整个 DB")

# ── 4. 执行 + 输出 ──
print(f"\n{'='*60}\n④ 依次执行 3 条 pending\n{'='*60}")

for pid in [pid_p1, pid_p2, pid_p3]:
    row = db.get(pid)
    agent = row["agent"] or "deepseek"
    model = row["model"] or MODEL
    prompt = {"agent": agent, "context": row["context"]}

    print(f"\n>>> 执行 #{pid} ...")
    print(f"  INPUT:  {json.dumps(prompt, ensure_ascii=False)}")

    try:
        result = resolve_prompt(prompt, db=db, model=model, timeout=180)
        print(f"  OUTPUT: {result}")
    except Exception as e:
        result = f"ERROR: {e}"
        print(f"  OUTPUT: {result}")

# ── 5. 读DB ──
dump("⑤ 执行后读取整个 DB")
