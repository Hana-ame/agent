import zlib

data = b"This is some data that will be compressed" * 100
compressed = zlib.compress(data)
print(f"Original size: {len(data)} bytes")
print(f"Zlib compressed size: {len(compressed)} bytes")
decompressed = zlib.decompress(compressed)
assert decompressed == data
print("Zlib decompressed successfully")

# 计算 CRC32
crc = zlib.crc32(data)
print(f"CRC32: {crc & 0xffffffff}")
