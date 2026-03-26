import bz2
import tempfile
import os

data = b"This is some data that will be compressed" * 100

with tempfile.NamedTemporaryFile() as tmp:
    with bz2.open(tmp.name, 'wb') as f:
        f.write(data)
    print(f"Original size: {len(data)} bytes")
    print(f"BZ2 compressed size: {os.path.getsize(tmp.name)} bytes")

    with bz2.open(tmp.name, 'rb') as f:
        decompressed = f.read()
    assert decompressed == data
    print("BZ2 decompressed successfully")
