import sys

# 列表推导式
list_comp = [x**2 for x in range(10000)]
# 生成器表达式
gen_exp = (x**2 for x in range(10000))

print(f"List comprehension size: {sys.getsizeof(list_comp)} bytes")
print(f"Generator expression size: {sys.getsizeof(gen_exp)} bytes")
print(f"First 5 from list: {list_comp[:5]}")
print(f"First 5 from generator: {[next(gen_exp) for _ in range(5)]}")
