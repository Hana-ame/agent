import zipfile
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    # 创建临时文件
    files = []
    for i in range(3):
        fpath = os.path.join(tmpdir, f"file{i}.txt")
        with open(fpath, 'w') as f:
            f.write(f"Content {i}")
        files.append(fpath)
    
    # 压缩
    zip_path = os.path.join(tmpdir, "archive.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for f in files:
            zf.write(f, os.path.basename(f))
    
    # 列出内容
    with zipfile.ZipFile(zip_path, 'r') as zf:
        print("Zip contents:", zf.namelist())
        # 读取一个文件
        with zf.open('file0.txt') as f:
            print("file0.txt content:", f.read().decode())
