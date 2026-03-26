from typing import TypeVar, List

T = TypeVar('T', int, float, str)

def first(items: List[T]) -> T:
    return items[0]

def repeat(item: T, times: int) -> List[T]:
    return [item] * times

if __name__ == "__main__":
    print(first([1, 2, 3]))
    print(first(["a", "b"]))
    print(repeat("x", 3))
    print(repeat(5.5, 2))
