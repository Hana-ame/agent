import heapq

data = [5, 2, 8, 1, 9, 3]
heapq.heapify(data)
print("Heap:", data)

heapq.heappush(data, 4)
print("After push 4:", data)

print("Pop smallest:", heapq.heappop(data))

# 获取 n 个最大/最小
print("3 largest:", heapq.nlargest(3, data))
print("2 smallest:", heapq.nsmallest(2, data))

# 合并多个有序列表
a = [1, 3, 5]
b = [2, 4, 6]
merged = list(heapq.merge(a, b))
print("Merged:", merged)
