"""Context 变异守护进程

读取已有记录的 context，对其进行变异：
- 数字 (ID): 替换为其他 ID
- 文本: 随机修改（添加、删除、替换字符）
- 列表: 添加、删除、换位

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


def get_all_ids():
    """获取所有记录的 id。"""
    rows = db.list_all()
    return [r["id"] for r in rows]


def mutate_id(item, all_ids):
    """变异一个 ID：替换为其他 ID。"""
    if not all_ids:
        return item
    # 排除自身，随机选一个
    others = [i for i in all_ids if i != item]
    if not others:
        return item
    return random.choice(others)


def mutate_text(text):
    """变异一段文本。"""
    if not text:
        return text

    mutation = random.choice(["insert", "delete", "replace", "swap"])

    if mutation == "insert":
        # 随机插入一个字符
        pos = random.randint(0, len(text))
        char = random.choice(" abcdefg一三五七九")
        return text[:pos] + char + text[pos:]

    elif mutation == "delete":
        # 随机删除一个字符
        if len(text) <= 1:
            return text
        pos = random.randint(0, len(text) - 1)
        return text[:pos] + text[pos + 1:]

    elif mutation == "replace":
        # 随机替换一个字符
        if not text:
            return text
        pos = random.randint(0, len(text) - 1)
        char = random.choice("0123456789一二三")
        return text[:pos] + char + text[pos + 1:]

    elif mutation == "swap":
        # 随机交换两个字符
        if len(text) < 2:
            return text
        i, j = random.sample(range(len(text)), 2)
        lst = list(text)
        lst[i], lst[j] = lst[j], lst[i]
        return "".join(lst)

    return text


def mutate_item(item, all_ids):
    """变异一个元素。"""
    if isinstance(item, int):
        return mutate_id(item, all_ids)
    elif isinstance(item, str):
        return mutate_text(item)
    return item


def mutate_list(items, all_ids):
    """对列表进行结构变异。"""
    mutation = random.choice(["add", "remove", "swap", "reverse"])

    if mutation == "add":
        new_id = random.choice(all_ids) if all_ids else None
        if new_id is not None:
            pos = random.randint(0, len(items))
            return items[:pos] + [new_id] + items[pos:]

    elif mutation == "remove":
        if len(items) > 1:
            pos = random.randint(0, len(items) - 1)
            return items[:pos] + items[pos + 1:]

    elif mutation == "swap":
        if len(items) >= 2:
            i, j = random.sample(range(len(items)), 2)
            items = items[:]
            items[i], items[j] = items[j], items[i]
            return items

    elif mutation == "reverse":
        return items[::-1]

    return items


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

        # 先对列表结构变异
        new_items = mutate_list(items, all_ids)

        # 再对每个元素变异
        final_items = []
        for item in new_items:
            if random.random() < 0.3:  # 30% 概率变异每个元素
                final_items.append(mutate_item(item, all_ids))
            else:
                final_items.append(item)

        # 跳过和原来一样的
        if final_items == items:
            continue

        # 添加新记录
        pid = db.add(
            final_items,
            agent=row["agent"],
            model=row["model"],
        )
        print(f"  #{row['id']} → #{pid}: {items} → {final_items}")
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
