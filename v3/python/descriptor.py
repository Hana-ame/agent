class PositiveNumber:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value):
        if value <= 0:
            raise ValueError(f"{self.name} must be positive")
        obj.__dict__[self.name] = value

class Person:
    age = PositiveNumber()

if __name__ == "__main__":
    p = Person()
    p.age = 30
    print(p.age)
    try:
        p.age = -5
    except ValueError as e:
        print("Error:", e)
