from dataclasses import dataclass

@dataclass(slots=True)
class Point:
    x: int
    y: int

p = Point(10, 20)
print(p)
try:
    p.z = 30
except AttributeError as e:
    print("Cannot add attribute:", e)
