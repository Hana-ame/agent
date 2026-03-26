# 列表推导与字典操作练习
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 列表推导：偶数平方
even_squares = [x**2 for x in numbers if x % 2 == 0]
print(f"Even numbers squares: {even_squares}")

# 字典推导：数字映射到平方
squares_dict = {x: x**2 for x in numbers}
print(f"Squares dict: {squares_dict}")

# 集合推导：大于5的数
greater_than_5 = {x for x in numbers if x > 5}
print(f"Numbers > 5: {greater_than_5}")

# 使用enumerate
print("Index-value pairs:")
for idx, val in enumerate(numbers[:5]):
    print(f"  Index {idx}: {val}")

# zip 使用
names = ["Alice", "Bob", "Charlie"]
scores = [95, 87, 91]
for name, score in zip(names, scores):
    print(f"{name} scored {score}")
