from functools import singledispatch

@singledispatch
def process(arg):
    return f"Default: {arg}"

@process.register(int)
def _(arg):
    return f"Integer: {arg}"

@process.register(list)
def _(arg):
    return f"List length: {len(arg)}"

if __name__ == "__main__":
    print(process("hello"))
    print(process(42))
    print(process([1,2,3]))
