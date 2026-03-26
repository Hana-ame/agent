import functools

# lru_cache 详细示例
@functools.lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print("fib(35):", fibonacci(35))
print("Cache info:", fibonacci.cache_info())

# total_ordering
@functools.total_ordering
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def __eq__(self, other):
        return self.age == other.age
    def __lt__(self, other):
        return self.age < other.age

p1 = Person("Alice", 30)
p2 = Person("Bob", 25)
print(f"p1 > p2: {p1 > p2}")
print(f"p1 <= p2: {p1 <= p2}")
