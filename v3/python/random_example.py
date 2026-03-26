import random

# 设置种子以重现
random.seed(42)

# 随机整数
print("Random int (1-10):", random.randint(1, 10))

# 随机浮点数
print("Random float (0-1):", random.random())

# 从列表随机选择
colors = ['red', 'green', 'blue']
print("Random choice:", random.choice(colors))

# 打乱列表
cards = list(range(1, 11))
random.shuffle(cards)
print("Shuffled cards:", cards)

# 随机采样
print("Sample 3 from 1-10:", random.sample(range(1, 11), 3))
