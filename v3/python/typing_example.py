from typing import List, Tuple

def greet(name: str) -> str:
    return f"Hello, {name}!"

def sum_and_product(numbers: List[int]) -> Tuple[int, int]:
    total = sum(numbers)
    product = 1
    for n in numbers:
        product *= n
    return total, product

if __name__ == "__main__":
    print(greet("Alice"))
    s, p = sum_and_product([1, 2, 3, 4])
    print(f"Sum: {s}, Product: {p}")
