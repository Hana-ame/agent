import gzip
import tempfile

data = b"This is some data that will be compressed" * 100

with tempfile.NamedTemporaryFile() as tmp:
    # 压缩写入
    with gzip.open(tmp.name, 'wb') as f:
        f.write(data)
    print(f"Original size: {len(data)} bytes")
    print(f"Compressed size: {os.path.getsize(tmp.name)} bytes")

    # 读取解压
    with gzip.open(tmp.name, 'rb') as f:
        decompressed = f.read()
    assert decompressed == data
    print("Decompressed successfully")
