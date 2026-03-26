from dataclasses import dataclass
from datetime import datetime

@dataclass
class Person:
    name: str
    birth_year: int

    @property
    def age(self):
        return datetime.now().year - self.birth_year

if __name__ == "__main__":
    p = Person("Alice", 1995)
    print(f"{p.name} is {p.age} years old")
