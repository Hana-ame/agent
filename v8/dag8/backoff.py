"""抖动指数退避：随机 429 下避免所有重试同步撞车。"""
from __future__ import annotations

import random
from typing import Callable, Optional

__all__ = ["jittered"]


def jittered(base: float, attempt: int, *, factor: float = 2.0,
             cap: float = 60.0,
             rand: Optional[Callable[[], float]] = None) -> float:
    """返回 base*factor**attempt 的 0.5~1.5 倍（上限 cap）。"""
    raw = min(cap, max(0.0, float(base)) * (float(factor) ** int(attempt)))
    r = rand() if rand is not None else random.random()
    return raw * (0.5 + float(r))
