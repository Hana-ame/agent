from functools import partialmethod

class Cell:
    def __init__(self, default):
        self.default = default

    def get(self, key, default):
        return self.default

    get_default = partialmethod(get, default=None)

c = Cell(42)
print(c.get_default('key'))   # 调用 get(self, 'key', default=None)
print(c.get('key', 100))
