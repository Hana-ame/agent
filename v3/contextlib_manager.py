import contextlib

@contextlib.contextmanager
def managed_resource():
    print("Acquiring resource")
    try:
        yield "resource"
    finally:
        print("Releasing resource")

with managed_resource() as res:
    print(f"Using {res}")

# 嵌套管理
@contextlib.contextmanager
def log_duration(name):
    import time
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"{name} took {elapsed:.3f}s")

with log_duration("sleep"):
    import time
    time.sleep(0.2)
