import itertools

# groupby - 分组
data = [('a', 1), ('a', 2), ('b', 3), ('b', 4), ('c', 5)]
for key, group in itertools.groupby(data, key=lambda x: x[0]):
    print(f"{key}: {list(group)}")

# accumulate - 累积计算
numbers = [1, 2, 3, 4, 5]
print("Accumulated sum:", list(itertools.accumulate(numbers)))
print("Accumulated product:", list(itertools.accumulate(numbers, lambda x,y: x*y)))

# chain - 连接多个可迭代对象
print("Chain:", list(itertools.chain([1,2], ['a','b'], (3,4))))

# combinations_with_replacement
print("Combinations with replacement:", list(itertools.combinations_with_replacement('ABC', 2)))
