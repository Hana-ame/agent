from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

if __name__ == "__main__":
    r = Rectangle(3, 4)
    print(f"Rectangle area: {r.area()}")
    # 尝试实例化 Shape 会出错
    try:
        s = Shape()
    except TypeError as e:
        print(f"Cannot instantiate abstract class: {e}")
