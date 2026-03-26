import shutil
import tempfile
import os

# 创建临时目录
with tempfile.TemporaryDirectory() as tmpdir:
    src = os.path.join(tmpdir, "source.txt")
    dst = os.path.join(tmpdir, "dest.txt")
    
    with open(src, 'w') as f:
        f.write("hello world")
    
    # 复制文件
    shutil.copy2(src, dst)
    print(f"Copied: {os.path.exists(dst)}")
    
    # 压缩目录
    archive = shutil.make_archive(os.path.join(tmpdir, "archive"), 'zip', tmpdir)
    print(f"Created archive: {archive}")
    
    # 解压
    extract_dir = os.path.join(tmpdir, "extracted")
    shutil.unpack_archive(archive, extract_dir)
    print(f"Extracted: {os.path.exists(extract_dir)}")
