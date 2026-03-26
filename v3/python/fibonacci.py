from __future__ import print_function

def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a)
        a, b = b, a + b

if __name__ == "__main__":
    fibonacci(10)
