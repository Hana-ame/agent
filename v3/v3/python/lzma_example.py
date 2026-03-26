import lzma
import tempfile
import os

data = b"This is some data that will be compressed" * 100

with tempfile.NamedTemporaryFile() as tmp:
    with lzma.open(tmp.name, 'wb') as f:
        f.write(data)
    print(f"Original size: {len(data)} bytes")
    print(f"LZMA compressed size: {os.path.getsize(tmp.name)} bytes")

    with lzma.open(tmp.name, 'rb') as f:
        decompressed = f.read()
    assert decompressed == data
    print("LZMA decompressed successfully")
