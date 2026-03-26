import contextlib
import sys
import io

# suppress 忽略特定异常
with contextlib.suppress(FileNotFoundError):
    open('nonexistent.txt').read()

# closing 自动调用 close
class Resource:
    def close(self):
        print("Resource closed")
    def use(self):
        print("Using resource")

with contextlib.closing(Resource()) as r:
    r.use()

# redirect_stdout 捕获输出
f = io.StringIO()
with contextlib.redirect_stdout(f):
    print("Hello from redirect")
output = f.getvalue()
print("Captured:", output.strip())
