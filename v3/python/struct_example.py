import struct

# 打包: 整数、浮点数、字符串
packed = struct.pack('i f 4s', 42, 3.14, b'data')
print("Packed:", packed)

# 解包
unpacked = struct.unpack('i f 4s', packed)
print("Unpacked:", unpacked)
