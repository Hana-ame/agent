from typing import Protocol, TypedDict, TypeGuard

class Greetable(Protocol):
    def greet(self) -> str: ...

class Person:
    def greet(self) -> str:
        return "Hello, I'm a person"

class Robot:
    def greet(self) -> str:
        return "Beep boop"

def make_greet(g: Greetable) -> None:
    print(g.greet())

class UserDict(TypedDict):
    name: str
    age: int

def process_user(data: UserDict) -> None:
    print(f"User {data['name']} is {data['age']} years old")

def is_string_list(obj) -> TypeGuard[list[str]]:
    return isinstance(obj, list) and all(isinstance(x, str) for x in obj)

if __name__ == "__main__":
    make_greet(Person())
    make_greet(Robot())
    process_user({"name": "Alice", "age": 30})
    print(is_string_list(["a", "b"]))
    print(is_string_list([1, 2]))
