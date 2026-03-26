class DynamicObject:
    def __init__(self):
        self._data = {}

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == '_data':
            super().__setattr__(name, value)
        else:
            self._data[name] = value

if __name__ == "__main__":
    obj = DynamicObject()
    obj.name = "Alice"
    obj.age = 30
    print(obj.name, obj.age)
    try:
        print(obj.city)
    except AttributeError as e:
        print("Error:", e)
