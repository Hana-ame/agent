import array

# 创建整数数组
arr = array.array('i', [1, 2, 3, 4, 5])
print("Array:", arr)
print("Item size:", arr.itemsize)
print("Memory usage:", arr.buffer_info()[1] * arr.itemsize, "bytes")
