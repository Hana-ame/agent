import functools
import time
import random

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt+1} failed: {e}, retrying in {delay}s")
                    time.sleep(delay)
        return wrapper
    return decorator

class CountCalls:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        print(f"Call {self.calls} of {self.func.__name__}")
        return self.func(*args, **kwargs)

@CountCalls
@retry(max_attempts=2, delay=0.1)
def unstable_function(fail_prob=0.5):
    if random.random() < fail_prob:
        raise ValueError("Random failure")
    return "Success"

if __name__ == "__main__":
    for _ in range(3):
        try:
            result = unstable_function(fail_prob=0.7)
            print("Result:", result)
        except ValueError as e:
            print("Final failure:", e)
