from dataclasses import dataclass, field
import random

@dataclass
class Student:
    name: str
    age: int
    id: int = field(init=False, repr=False)
    
    def __post_init__(self):
        self.id = random.randint(1000, 9999)

s = Student("Bob", 20)
print(s)
print(f"ID: {s.id}")
