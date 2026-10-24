"""RPM 限速器：让调度器在窗口内最多发起 `rpm` 次 API 调用。

背景：v8 的波次并行 + pi 子进程 + v7 直连会同时打到同一个网关，
429（tpm/rpm limit）是实测里最大的噪声源——GA 里一半失败都是它。
节点重试只能事后补救；在调度层提前节流更便宜。

只对「会打 API 的节点」生效（pi / graph），command 节点不计。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Optional

__all__ = ["RateLimiter", "shared_limiter"]

_LIMITERS: dict[tuple, RateLimiter] = {}


def shared_limiter(rpm: int, window: float = 60.0) -> RateLimiter:
    """进程级共享：同一 rpm/window 的所有 run、judge 共用一份配额。"""
    key = (int(rpm), float(window))
    if key not in _LIMITERS:
        _LIMITERS[key] = RateLimiter(rpm, window)
    return _LIMITERS[key]


class RateLimiter:
    def __init__(self, rpm: Optional[int], window: float = 60.0) -> None:
        self.rpm = int(rpm) if rpm else 0
        self.window = float(window)
        self._starts: Deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """阻塞到窗口内有空位，并登记本次调用。"""
        if self.rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._starts and now - self._starts[0] >= self.window:
                    self._starts.popleft()
                if len(self._starts) < self.rpm:
                    self._starts.append(now)
                    return
                wait = self.window - (now - self._starts[0]) + 0.01
            time.sleep(max(wait, 0.01))
