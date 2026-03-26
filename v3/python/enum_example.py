from enum import Enum, auto

class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()

print("Color.RED:", Color.RED)
print("Color.RED name:", Color.RED.name)
print("Color.RED value:", Color.RED.value)

# 迭代所有成员
for color in Color:
    print(color)
