import functools
import operator

numbers = [1, 2, 3, 4, 5]

# 求和
total = functools.reduce(operator.add, numbers)
print(f"Sum: {total}")

# 乘积
product = functools.reduce(operator.mul, numbers)
print(f"Product: {product}")

# 字符串连接
words = ["Hello", " ", "World", "!"]
sentence = functools.reduce(operator.add, words)
print(f"Concatenated: {sentence}")
