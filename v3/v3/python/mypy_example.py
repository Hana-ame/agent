def add(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> str:
    return f"Hello, {name}"

if __name__ == "__main__":
    result = add(5, 3)
    print(f"5+3={result}")
    print(greet("World"))
