import random
import math
from typing import List, Tuple


def calculate_fitness(quality: float, cost: float, time_ms: int,
                      w1: float = 10.0, w2: float = 1.0, w3: float = 0.1) -> float:
    return w1 * quality - w2 * math.log(1 + cost) - w3 * math.log(1 + time_ms / 1000)


def selection(scores: List[Tuple[int, float]], top_pct: float = 0.3) -> List[int]:
    sorted_items = sorted(scores, key=lambda x: x[1], reverse=True)
    count = max(1, int(len(sorted_items) * top_pct))
    return [item[0] for item in sorted_items[:count]]


def mutate(template: str) -> str:
    mutations = [
        ("。", "。请仔细思考后再回答。"),
        ("输出", "请确保输出准确，"),
        ("请", "请仔细地"),
        ("。", "。确保代码质量。"),
    ]
    if random.random() < 0.5:
        old, new = random.choice(mutations)
        return template.replace(old, new, 1)
    return template


def crossover(parent1: str, parent2: str) -> str:
    mid1 = len(parent1) // 2
    mid2 = len(parent2) // 2
    return parent1[:mid1] + parent2[mid2:]
