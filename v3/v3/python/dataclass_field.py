from dataclasses import dataclass, field
import random

def random_id():
    return random.randint(1000, 9999)

@dataclass
class Product:
    name: str
    price: float
    id: int = field(default_factory=random_id)
    tags: list = field(default_factory=list)

p1 = Product("Laptop", 999.99)
p2 = Product("Mouse", 29.99)
print(p1)
print(p2)
print("Tags default empty:", p1.tags)
