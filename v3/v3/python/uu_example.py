import uu
import tempfile
import os

data = b"This is some data for uu encoding"

with tempfile.NamedTemporaryFile() as tmp_in, tempfile.NamedTemporaryFile() as tmp_out:
    tmp_in.write(data)
    tmp_in.flush()
    
    # 编码
    uu.encode(tmp_in.name, tmp_out.name, b"test.txt")
    tmp_out.seek(0)
    encoded = tmp_out.read()
    print(f"Encoded data (first 100): {encoded[:100]}")
    
    # 解码
    with tempfile.NamedTemporaryFile() as decoded_file:
        uu.decode(tmp_out.name, decoded_file.name)
        decoded_file.seek(0)
        decoded = decoded_file.read()
        print(f"Decoded: {decoded}")
        assert data == decoded
