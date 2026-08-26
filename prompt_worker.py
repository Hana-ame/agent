"""Prompt 处理守护进程

循环读取 pending 记录，随机取一条执行，更新结果。

用法:  python3 prompt_worker.py
"""

import sys
import time
import signal
import random

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


def process_one_pending():
    """随机取一条 pending 记录并执行。"""
    rows = db.list_by_status("pending")
    if not rows:
        return None

    target = random.choice(rows)
    pid = target["id"]
    agent = target["agent"] or "Null"
    model = target["model"] or "siliconflow-cn/Qwen/Qwen3-8B"

    print(f"  处理 #{pid}: context={target['context'][:60]}...")

    prompt = {"agent": agent, "context": target["context"]}

    try:
        result = resolve_prompt(prompt, db=db, model=model, timeout=300)
        print(f"  #{pid} 完成: {result[:80]}...")
        return pid
    except Exception as e:
        print(f"  #{pid} 失败: {e}")
        db.failed(pid, str(e))
        return None


def main():
    print("=" * 50)
    print("  Prompt Worker 守护进程")
    print("  按 Ctrl+C 退出")
    print("=" * 50)

    cycle = 0
    while running:
        cycle += 1
        processed = process_one_pending()

        if processed is not None:
            print(f"  [周期 {cycle}] 已处理 #{processed}")
        else:
            if cycle % 10 == 1:
                print(f"  [周期 {cycle}] 无 pending 记录，等待中...")

        time.sleep(2)

    print("Worker 已退出")


if __name__ == "__main__":
    main()
