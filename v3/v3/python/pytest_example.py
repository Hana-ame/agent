try:
    import pytest
except ImportError:
    print("pytest not installed, skipping")
    exit(0)

def add(a, b):
    return a + b

# 伪测试（仅演示）
print("pytest is available. To run tests, use 'pytest' command.")
print("Example test function:")
print("def test_add(): assert add(1,2) == 3")
