class MetaLogger(type):
    def __new__(cls, name, bases, dct):
        print(f"Creating class {name}")
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=MetaLogger):
    pass

if __name__ == "__main__":
    print("Class MyClass created")
    obj = MyClass()
    print("Instance created")
