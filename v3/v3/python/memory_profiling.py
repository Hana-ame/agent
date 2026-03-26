"""简单内存使用示例（使用sys.getsizeof）"""
import sys

data = [i for i in range(1000)]
print(f"List of 1000 ints: {sys.getsizeof(data)} bytes")
print(f"Each int: {sys.getsizeof(0)} bytes")
print(f"Total estimate: {sys.getsizeof(data) + len(data)*sys.getsizeof(0)} bytes")
