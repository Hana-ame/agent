import array

# 创建数组
arr = array.array('i', range(10))
print("Original:", arr)

# 内存视图
mv = memoryview(arr)
print("Memoryview:", mv.tolist())

# 通过 memoryview 修改数据
mv[2] = 99
print("After modification:", arr)

# 切片视图
slice_mv = mv[5:8]
slice_mv[1] = 100
print("After slice modification:", arr)
