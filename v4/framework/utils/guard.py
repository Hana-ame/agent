import operator
import re
from dataclasses import dataclass
from typing import Any, Optional

OPERATORS = {
    ">=": operator.ge, ">": operator.gt,
    "<=": operator.le, "<": operator.lt,
    "==": operator.eq, "!=": operator.ne,
    "in": lambda a, b: a in b,
    "contains": lambda a, b: b in a,
    "matches": lambda a, b: re.match(b, str(a)) is not None,
}

@dataclass(frozen=True)
class Guard:
    field: Optional[str] = None
    op: str = "=="
    value: Any = None
    match: Optional[dict] = None
    mode: str = "single"
    guards: Optional[list] = None

    def evaluate(self, data: Any) -> bool:
        if self.mode == "all":
            return all(g.evaluate(data) for g in (self.guards or []))
        if self.mode == "any":
            return any(g.evaluate(data) for g in (self.guards or []))
            
        if self.match is not None:
            return isinstance(data, dict) and all(
                data.get(k) == v for k, v in self.match.items()
            )
        
        target = data
        if self.field and isinstance(data, dict):
            target = data.get(self.field)
        
        fn = OPERATORS.get(self.op)
        if fn is None:
            raise ValueError(f"Unknown guard operator: {self.op}")
        return fn(target, self.value)

    def __and__(self, other: "Guard") -> "Guard":
        return Guard(mode="all", guards=[self, other])
    
    def __or__(self, other: "Guard") -> "Guard":
        return Guard(mode="any", guards=[self, other])

def build_guard(config) -> Guard:
    if isinstance(config, list):
        return Guard(mode="all", guards=[build_guard(c) for c in config])
    return Guard(**config)
