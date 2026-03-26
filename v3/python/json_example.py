import json

data = {
    "name": "Alice",
    "age": 30,
    "city": "New York",
    "hobbies": ["reading", "coding", "hiking"],
    "is_student": False
}

# 写入 JSON 文件
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

# 读取 JSON 文件
with open("data.json", "r") as f:
    loaded_data = json.load(f)

print("Original:", data)
print("Loaded:  ", loaded_data)
assert data == loaded_data

import os
os.remove("data.json")
