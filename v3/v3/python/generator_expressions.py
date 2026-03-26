"""生成器表达式 vs 列表推导性能对比"""
import sys
import time

# 列表推导
start = time.time()
list_comp = [x**2 for x in range(10000)]
list_time = time.time() - start

# 生成器表达式
start = time.time()
gen_exp = (x**2 for x in range(10000))
gen_time = time.time() - start

print(f"List comprehension time: {list_time:.5f}s, size: {sys.getsizeof(list_comp)} bytes")
print(f"Generator expression time: {gen_time:.5f}s, size: {sys.getsizeof(gen_exp)} bytes")
print(f"First 5 from list: {list_comp[:5]}")
print(f"First 5 from generator: {[next(gen_exp) for _ in range(5)]}")
