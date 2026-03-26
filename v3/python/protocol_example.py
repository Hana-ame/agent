from typing import Protocol

class Greeter(Protocol):
    def greet(self) -> str:
        ...

class Person:
    def greet(self) -> str:
        return "Hello from Person"

class Robot:
    def greet(self) -> str:
        return "Beep boop"

def say_hello(g: Greeter) -> None:
    print(g.greet())

if __name__ == "__main__":
    say_hello(Person())
    say_hello(Robot())
