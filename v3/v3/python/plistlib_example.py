import plistlib
import tempfile
import os

data = {
    'name': 'Alice',
    'age': 30,
    'hobbies': ['reading', 'coding']
}

with tempfile.NamedTemporaryFile(suffix='.plist') as tmp:
    # 写入二进制 plist
    with open(tmp.name, 'wb') as f:
        plistlib.dump(data, f)
    
    # 读取
    with open(tmp.name, 'rb') as f:
        loaded = plistlib.load(f)
    print("Loaded plist:", loaded)
