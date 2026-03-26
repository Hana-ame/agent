# 创建 bytearray
ba = bytearray(b'hello world')
print("Original:", ba)

# 创建 memoryview
mv = memoryview(ba)
print("Memoryview:", mv.tolist())

# 通过 memoryview 修改
mv[6:11] = b'python'
print("After modification:", ba)

# 切片视图
slice_view = mv[0:5]
slice_view[:] = b'HELLO'
print("After slice modification:", ba)
