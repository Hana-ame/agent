from typing import TypedDict, Literal, Optional

class Person(TypedDict):
    name: str
    age: int
    city: Optional[str]

def greet(person: Person) -> str:
    return f"Hello, {person['name']}!"

def set_status(status: Literal['active', 'inactive']) -> str:
    return f"Status set to {status}"

if __name__ == "__main__":
    alice: Person = {"name": "Alice", "age": 30, "city": "New York"}
    print(greet(alice))
    print(set_status('active'))
