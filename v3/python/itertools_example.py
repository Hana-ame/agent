import itertools

# 无限循环
print("Cycle (first 5):", list(itertools.islice(itertools.cycle(['A', 'B', 'C']), 5)))

# 排列组合
items = [1, 2, 3]
print("Permutations:", list(itertools.permutations(items, 2)))
print("Combinations:", list(itertools.combinations(items, 2)))
print("Product:", list(itertools.product([1, 2], ['a', 'b'])))
