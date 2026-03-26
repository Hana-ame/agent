import contextlib
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    filenames = [os.path.join(tmpdir, f"file{i}.txt") for i in range(3)]
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(open(fname, 'w')) for fname in filenames]
        for i, f in enumerate(files):
            f.write(f"Content {i}\n")
        print(f"Wrote to {len(files)} files")
    # 文件已关闭
    print("Files closed")
    # 验证内容
    for fname in filenames:
        with open(fname) as f:
            print(f"{fname}: {f.read().strip()}")
