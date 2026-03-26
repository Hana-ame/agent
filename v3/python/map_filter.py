from __future__ import print_function

# 计算 1-10 的平方
squares = list(map(lambda x: x**2, range(1, 11)))
print("Squares:", squares)

# 筛选出偶数平方
even_squares = list(filter(lambda x: x % 2 == 0, squares))
print("Even squares:", even_squares)

# 使用列表推导式实现同样的功能
even_squares_comp = [x**2 for x in range(1, 11) if x**2 % 2 == 0]
print("Even squares (list comp):", even_squares_comp)
