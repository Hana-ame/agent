import mmap
import tempfile
import os

with tempfile.NamedTemporaryFile() as tmp:
    tmp.write(b"Hello mmap world!")
    tmp.flush()
    
    with mmap.mmap(tmp.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        print("Content via mmap:", mm.readline().decode().strip())
        # 修改内存
        mm.seek(6)
        mm.write(b"PYTHON")
        mm.flush()
    
    # 验证文件修改
    with open(tmp.name, 'rb') as f:
        print("File content after mmap:", f.read())
