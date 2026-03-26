from functools import singledispatchmethod

class Processor:
    @singledispatchmethod
    def process(self, arg):
        return f"Default: {arg}"

    @process.register(int)
    def _(self, arg):
        return f"Integer: {arg}"

    @process.register(str)
    def _(self, arg):
        return f"String: {arg}"

p = Processor()
print(p.process(42))
print(p.process("hello"))
print(p.process([1,2]))
