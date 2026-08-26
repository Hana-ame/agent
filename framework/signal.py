"""Abort signal - 在边之间透传的中止信号。

中止信号(AbortSignal)：当某条边判断"没有必要继续"时(例如 agent 筛选出的
数据数量不足、某个前置条件不满足)，它可以把一个 AbortSignal + reason 写入
目标顶点，后续所有边检测到该信号后不再执行正常逻辑，而是直接把信号
继续透传给下游顶点，直到到达最终顶点，由调用方读取 reason。

设计要点：
    - AbortSignal 是一个普通 Python 对象，可像数据一样被 get/set 传递；
    - 携带 reason(人类可读的原因) 与 source(产生该信号的边 ID)；
    - edge 通过框架提供的 Edge.forward_abort() 一键透传。
"""

from typing import Optional


class AbortSignal:
    """中止信号：携带 reason 与产生源，在边之间透传。"""

    __slots__ = ("reason", "source")

    def __init__(self, reason: str, source: Optional[str] = None):
        self.reason = reason
        self.source = source

    def __repr__(self) -> str:
        src = f", source={self.source!r}" if self.source else ""
        return f"AbortSignal(reason={self.reason!r}{src})"

    def __str__(self) -> str:
        return f"[ABORTED] {self.reason}"


def is_abort(value) -> bool:
    """判断一个值是否为 Abort 信号。"""
    return isinstance(value, AbortSignal)


def abort_reason(value) -> Optional[str]:
    """若 value 是 Abort 信号，返回其 reason；否则返回 None。"""
    return value.reason if isinstance(value, AbortSignal) else None
