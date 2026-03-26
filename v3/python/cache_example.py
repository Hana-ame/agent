import functools
import time

@functools.cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

start = time.time()
fibonacci(35)
print(f"Time: {time.time() - start:.4f}s")
print(f"fib(35) = {fibonacci(35)}")
