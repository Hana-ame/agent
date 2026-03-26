"""使用contextlib.ExitStack管理多个上下文"""
import contextlib
import tempfile

with contextlib.ExitStack() as stack:
    files = [stack.enter_context(tempfile.NamedTemporaryFile(mode='w+')) for _ in range(3)]
    for i, f in enumerate(files):
        f.write(f"File {i} content")
        f.seek(0)
        print(f"File {i}: {f.read()}")
print("All files automatically closed")
