import inspect
import sys

def example_function(a, b=10, *args, **kwargs):
    """This is a docstring."""
    pass

# 获取函数签名
sig = inspect.signature(example_function)
print("Signature:", sig)
print("Parameters:", list(sig.parameters.keys()))

# 获取源代码
print("Source lines:", inspect.getsourcelines(example_function)[0][:3])

# 检查对象类型
print("Is function?", inspect.isfunction(example_function))
print("Is module?", inspect.ismodule(sys))
