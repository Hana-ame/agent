import itertools

# 无限循环取前10个
colors = itertools.cycle(['red', 'green', 'blue'])
first_10 = list(itertools.islice(colors, 10))
print("First 10 cycles:", first_10)

# 使用 cycle 实现轮询
def round_robin(*iterables):
    iters = [iter(it) for it in iterables]
    for val in itertools.cycle(iters):
        try:
            yield next(val)
        except StopIteration:
            # 移除已耗尽的迭代器
            iters.remove(val)

a = [1, 2, 3]
b = ['a', 'b']
c = [10, 20, 30, 40]
result = list(itertools.islice(round_robin(a, b, c), 10))
print("Round robin:", result)
