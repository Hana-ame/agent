"""Context 变异守护进程

读取已有记录的 context，对其进行变异（添加、删除、换位），
生成新变体并作为 pending 记录添加回数据库。

用法:  python3 prompt_mutator.py
"""

import sys
import json
import random
import time
import signal

sys.path.insert(0, ".")

from prompt_db import PromptDB

db = PromptDB()
running = True


def handle_signal(sig, frame):
    global running
    print("\n收到信号，正在退出...")
    running = False


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def parse_context(ctx_str):
    """解析 context 字符串为列表。"""
    try:
        ctx = json.loads(ctx_str)
        if isinstance(ctx, list):
            return ctx
        return [ctx]
    except (json.JSONDecodeError, TypeError):
        if ctx_str:
            return [ctx_str]
        return []


def mutate_add(items, all_ids):
    """随机添加一个 id。"""
    if not all_ids:
        return items
    new_id = random.choice(all_ids)
    pos = random.randint(0, len(items))
    return items[:pos] + [new_id] + items[pos:]


def mutate_remove(items):
    """随机删除一个元素。"""
    if len(items) <= 1:
        return items
    pos = random.randint(0, len(items) - 1)
    return items[:pos] + items[pos + 1:]


def mutate_swap(items):
    """随机交换两个元素的位置。"""
    if len(items) < 2:
        return items
    i, j = random.sample(range(len(items)), 2)
    items = items[:]
    items[i], items[j] = items[j], items[i]
    return items


def mutate_reverse(items):
    """反转整个列表。"""
    return items[::-1]


def get_all_ids():
    """获取所有记录的 id。"""
    rows = db.list_all()
    return [r["id"] for r in rows]


def process_mutations():
    """对数据库中的 context 进行变异。"""
    rows = db.list_all()
    all_ids = get_all_ids()

    if len(rows) < 2:
        return 0

    mutated = 0
    # 随机选择一些记录进行变异
    candidates = [r for r in rows if r["context"] and r["context"] != "[]"]
    if not candidates:
        return 0

    sample_size = min(3, len(candidates))
    sampled = random.sample(candidates, sample_size)

    for row in sampled:
        items = parse_context(row["context"])
        if not items:
            continue

        # 随机选择一种变异
        mutation = random.choice(["add", "remove", "swap", "reverse"])

        if mutation == "add":
            new_items = mutate_add(items, all_ids)
        elif mutation == "remove":
            new_items = mutate_remove(items)
        elif mutation == "swap":
            new_items = mutate_swap(items)
        elif mutation == "reverse":
            new_items = mutate_reverse(items)
        else:
            continue

        # 跳过和原来一样的
        if new_items == items:
            continue

        # 添加新记录
        pid = db.add(
            new_items,
            agent=row["agent"],
            model=row["model"],
        )
        print(f"  #{row['id']} → #{pid} ({mutation}): {items} → {new_items}")
        mutated += 1

    return mutated


def main():
    print("=" * 50)
    print("  Prompt Mutator 守护进程")
    print("  按 Ctrl+C 退出")
    print("=" * 50)

    cycle = 0
    while running:
        cycle += 1
        mutated = process_mutations()

        if mutated > 0:
            print(f"\n  [周期 {cycle}] 变异了 {mutated} 条记录")

        time.sleep(10)

    print("Mutator 已退出")


if __name__ == "__main__":
    main()
