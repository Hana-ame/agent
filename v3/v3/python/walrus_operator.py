import re

text = "The quick brown fox jumps over the lazy dog"

# 传统方式
m = re.search(r'fox', text)
if m:
    print(f"Found '{m.group()}'")

# 海象运算符
if (m := re.search(r'fox', text)):
    print(f"Found '{m.group()}' using walrus")

# 在列表推导中使用
numbers = [1, 2, 3, 4, 5]
squared_evens = [y for x in numbers if (y := x**2) % 2 == 0]
print("Squared evens:", squared_evens)
