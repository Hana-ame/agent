from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

def greet(greeting, name):
    return f"{greeting}, {name}!"

hello = partial(greet, "Hello")
hi = partial(greet, "Hi")

if __name__ == "__main__":
    print(square(5))
    print(cube(5))
    print(hello("Alice"))
    print(hi("Bob"))
