from dataclasses import dataclass
from typing import TypeVar, Generic

T = TypeVar("T")

@dataclass(frozen=True)
class Slot(Generic[T]):
    """Typed, named data slot that edges produce and consume."""
    name: str
    type: type
    description: str = ""
