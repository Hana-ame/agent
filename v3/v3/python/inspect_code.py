import inspect
import sys

def example(x: int, y: int = 10) -> int:
    """Add two numbers."""
    return x + y

sig = inspect.signature(example)
print("Signature:", sig)
print("Parameters:", list(sig.parameters.keys()))
print("Docstring:", inspect.getdoc(example))
print("Source:", inspect.getsource(example)[:100])
