import contextlib
import urllib.request

# suppress 忽略异常
with contextlib.suppress(FileNotFoundError):
    open('nonexistent.txt').read()

# closing 自动关闭
class Resource:
    def close(self):
        print("Resource closed")

with contextlib.closing(Resource()) as r:
    print("Using resource")
