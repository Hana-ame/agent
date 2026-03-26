import json
from datetime import datetime

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

data = {
    "name": "Alice",
    "timestamp": datetime.now()
}

json_str = json.dumps(data, cls=CustomEncoder, indent=2)
print("Custom JSON:", json_str)

# 解析回来
loaded = json.loads(json_str)
print("Parsed:", loaded)
