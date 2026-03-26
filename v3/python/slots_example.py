class Point:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(10, 20)
print(f"Point: ({p.x}, {p.y})")
# 尝试动态添加属性会失败
try:
    p.z = 30
except AttributeError as e:
    print("Error:", e)
