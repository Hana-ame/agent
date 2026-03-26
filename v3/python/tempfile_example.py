import tempfile
import os

# 临时文件（自动删除）
with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt') as tmp:
    tmp.write("temporary data")
    tmp.seek(0)
    print("Temp file content:", tmp.read())
    print("Temp file name:", tmp.name)
# 文件已删除

# 临时目录
with tempfile.TemporaryDirectory() as tmpdir:
    print("Temporary directory:", tmpdir)
    with open(os.path.join(tmpdir, "test.txt"), 'w') as f:
        f.write("data")
# 目录已删除
