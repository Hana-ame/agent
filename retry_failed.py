import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_db import PromptDB
from resolve_prompt import resolve_prompt

DEFAULT_MODEL = "siliconflow-cn/Qwen/Qwen3-8B"
TIMEOUT = 60


def run_one(pid, db):
    row = db.get(pid)
    if not row:
        return False

    model = DEFAULT_MODEL
    agent = row["agent"] or "Null"
    context = row["context"]
    prompt = {"agent": agent, "context": context}

    t0 = time.time()
    try:
        result = resolve_prompt(prompt, db=db, model=model, timeout=TIMEOUT)
        db.done(pid, result, {"source": "retry", "elapsed": round(time.time() - t0, 2)})
        print(f"  #{pid} OK [{time.time()-t0:.0f}s]")
        return True
    except Exception as e:
        print(f"  #{pid} FAIL [{time.time()-t0:.0f}s] {str(e)[:80]}")
        return False


def main():
    db = PromptDB()
    entries = db.list_by_status("failed")

    if not entries:
        print("No failed entries.")
        return

    print(f"Retrying {len(entries)} failed entries with {DEFAULT_MODEL}")

    ok = fail = 0
    for row in entries:
        pid = row["id"]
        print(f"--- #{pid} {row['context'][:50]}...")
        if run_one(pid, db):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} OK, {fail} failed")


if __name__ == "__main__":
    main()
