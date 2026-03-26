import inspect

def func_a():
    func_b()

def func_b():
    func_c()

def func_c():
    stack = inspect.stack()
    for frame_info in stack:
        print(f"{frame_info.function}:{frame_info.lineno}")

func_a()
