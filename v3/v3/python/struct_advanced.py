import struct

# 打包多个类型：整数、浮点数、字节数组
packed = struct.pack('>i f 4s', 100, 3.14, b'data')
print(f"Packed: {packed.hex()}")

# 解包
unpacked = struct.unpack('>i f 4s', packed)
print(f"Unpacked: {unpacked}")

# 处理网络字节顺序
data = struct.pack('!I', 0x12345678)
print(f"Network byte order: {data.hex()}")
print(f"Unpacked: {struct.unpack('!I', data)[0]:#x}")
