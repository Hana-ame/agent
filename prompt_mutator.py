"""Context 变异守护进程（纯 Python 算法驱动）

使用遗传算法（交叉 + 变异）自动生成新的 context，无需调用 LLM。
只对数组类型的 context 进行变异，新生成的 context 插入数据库后由 Worker 执行。

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


# ── 辅助函数 ──────────────────────────────────────────────

def parse_context(ctx_str):
    """安全解析 context 为列表，若解析失败则包装为单元素列表。"""
    try:
        ctx = json.loads(ctx_str)
        if isinstance(ctx, list):
            return ctx
        return [ctx]
    except (json.JSONDecodeError, TypeError):
        if ctx_str:
            return [ctx_str]
        return []


def validate_context(ctx):
    """验证 context 是否为合法的 JSON 列表，且元素只能是 str 或 int。"""
    if not isinstance(ctx, list) or len(ctx) == 0:
        return False
    return all(isinstance(x, (str, int)) for x in ctx)


# ── 变异策略参数 ──────────────────────────────────────────

# 预置的"基因片段"——变异时可能插入的新字符串
SEED_STRINGS = [
    "请总结以下内容",
    "用一句话回答",
    "用三个要点解释",
    "用代码示例说明",
    "翻译成英文",
    "解释为什么会这样",
    "请详细说明原因",
    "用简单的语言回答",
]

# 交叉概率
CROSSOVER_RATE = 0.8
# 每个元素发生变异的概率
MUTATION_RATE = 0.3
# 每次运行生成的新个体数量（可配置）
MUTANTS_PER_CYCLE = 2


# ── 遗传算法核心 ─────────────────────────────────────────

def _select_parents(population):
    """从种群中随机选择两个不同的父个体。"""
    return random.sample(population, 2)


def _crossover(parent1, parent2):
    """单点交叉：随机选择一个切割点，交换两个列表的片段。"""
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]

    if len(parent1) < 2 or len(parent2) < 2:
        return parent1[:], parent2[:]

    point1 = random.randint(1, len(parent1) - 1)
    point2 = random.randint(1, len(parent2) - 1)

    child1 = parent1[:point1] + parent2[point2:]
    child2 = parent2[:point2] + parent1[point1:]

    return child1, child2


def _mutate(individual, valid_ids):
    """
    对个体进行随机变异：
    - 替换某个元素为随机 id 或随机字符串
    - 插入一个新元素
    - 删除一个元素
    保证变异后个体非空且元素类型合法。
    """
    if not individual:
        return individual

    mutated = individual[:]
    action = random.choice(["replace", "insert", "delete"])

    if action == "replace" and len(mutated) > 0:
        idx = random.randrange(len(mutated))
        if random.random() < 0.5 and valid_ids:
            mutated[idx] = random.choice(valid_ids)
        else:
            mutated[idx] = random.choice(SEED_STRINGS)

    elif action == "insert":
        pos = random.randint(0, len(mutated))
        if random.random() < 0.5 and valid_ids:
            element = random.choice(valid_ids)
        else:
            element = random.choice(SEED_STRINGS)
        mutated.insert(pos, element)

    elif action == "delete" and len(mutated) > 1:
        idx = random.randrange(len(mutated))
        del mutated[idx]

    return mutated


def generate_mutant(parent1, parent2, valid_ids):
    """
    通过交叉+变异生成一个新个体。
    如果交叉/变异后非法，返回 None。
    """
    child1, child2 = _crossover(parent1, parent2)
    child = random.choice([child1, child2])

    for i in range(len(child)):
        if random.random() < MUTATION_RATE:
            if random.random() < 0.5 and valid_ids:
                child[i] = random.choice(valid_ids)
            else:
                child[i] = random.choice(SEED_STRINGS)

    if random.random() < 0.5:
        child = _mutate(child, valid_ids)

    child = [x for x in child if x != ""]

    if not validate_context(child):
        return None

    return child


# ── 主循环 ────────────────────────────────────────────────

def process_mutations():
    """
    从数据库中选取数组类型的 done 记录作为种群，
    通过遗传算法生成新 context，插入数据库（状态 pending）。
    """
    rows = db.list_all()
    done_rows = [r for r in rows if r["status"] == "done"]
    all_ids = [r["id"] for r in done_rows]

    population = []
    for row in done_rows:
        ctx_list = parse_context(row["context"])
        if isinstance(ctx_list, list) and len(ctx_list) > 0:
            population.append(ctx_list)

    if len(population) < 2:
        return 0

    mutated = 0
    for _ in range(MUTANTS_PER_CYCLE):
        if len(population) < 2:
            break

        parent1, parent2 = _select_parents(population)
        new_context = generate_mutant(parent1, parent2, all_ids)

        if new_context is None:
            continue

        if new_context in (parent1, parent2):
            continue

        sample_row = random.choice([r for r in done_rows if parse_context(r["context"]) in (parent1, parent2)])
        agent = sample_row["agent"] if sample_row else ""
        model = sample_row["model"] if sample_row else ""

        pid = db.add(new_context, agent=agent, model=model)
        print(f"  → 新增变异体 #{pid}: {new_context}")
        mutated += 1

    return mutated


def main():
    print("=" * 50)
    print("  Prompt Mutator (遗传算法版)")
    print("  按 Ctrl+C 退出")
    print("=" * 50)

    cycle = 0
    while running:
        cycle += 1
        mutated = process_mutations()

        if mutated > 0:
            print(f"\n  [周期 {cycle}] 生成了 {mutated} 条新记录")

        time.sleep(30)

    print("Mutator 已退出")


if __name__ == "__main__":
    main()
