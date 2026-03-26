from typing import Final, TypeAlias

# 常量类型
MAX_SIZE: Final = 100

# 类型别名
Vector: TypeAlias = list[float]

def scale(v: Vector, factor: float) -> Vector:
    return [x * factor for x in v]

if __name__ == "__main__":
    v: Vector = [1.0, 2.0, 3.0]
    result = scale(v, 2.0)
    print(f"Original: {v}, scaled: {result}")
    print(f"MAX_SIZE = {MAX_SIZE}")
