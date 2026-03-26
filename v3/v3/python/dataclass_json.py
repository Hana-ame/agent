import json
from dataclasses import dataclass, asdict

@dataclass
class User:
    name: str
    age: int
    active: bool = True

user = User("Alice", 30)
json_str = json.dumps(asdict(user))
print("JSON:", json_str)
loaded = User(**json.loads(json_str))
print("Loaded:", loaded)
