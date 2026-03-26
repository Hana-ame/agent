import ctypes
import sys

# 获取 libc
if sys.platform == 'win32':
    libc = ctypes.CDLL('msvcrt')
else:
    libc = ctypes.CDLL('libc.so.6')

# getpid 通常存在
getpid = libc.getpid
getpid.restype = ctypes.c_int

pid = getpid()
print(f"Process ID: {pid}")

# 调用系统时间函数
if hasattr(libc, 'time'):
    time = libc.time
    time.restype = ctypes.c_long
    t = time(ctypes.c_void_p())
    print(f"Current time: {t}")
else:
    print("time function not available")
