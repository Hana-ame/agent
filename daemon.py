"""
异步调用 opencode，处理 pending 记录。
同时调用 LLM 生成新的 context 变体。

用法:
    python3 daemon.py
"""

import sys
import time
import signal

sys.path.insert(0, ".")

from prompt_db import PromptDB
from resolve_prompt import resolve_prompt

db = PromptDB()
running = True


def handle_signal(sig, frame):
    global running
    print("\n收到信号，正在退出...")
    running = False


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def process_pending():
    rows = db.list_by_status("pending")
    if not rows:
        return 0

    processed = 0
    for row in rows:
        pid = row["id"]
        context = row["context"]
        model = row["model"]

        print(f"  处理 #{pid}: context={context[:60]}...")
        try:
            result = resolve_prompt(pid, db=db, model=model, timeout=300)
            print(f"  #{pid} 完成: {result[:80]}...")
            processed += 1
        except Exception as e:
            print(f"  #{pid} 失败: {e}")
            db.failed(pid, str(e))

    return processed


def main():
    print("=" * 50)
    print("  Prompt Daemon")
    print("  按 Ctrl+C 退出")
    print("=" * 50)

    cycle = 0
    while running:
        cycle += 1
        processed = process_pending()
        if processed > 0:
            print(f"\n  [周期 {cycle}] 处理了 {processed} 条记录")

        time.sleep(2)

    print("Daemon 已退出")


if __name__ == "__main__":
    main()