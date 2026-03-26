from enum import Enum, auto, unique

@unique
class Status(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()

print("Status values:")
for status in Status:
    print(f"{status.name} = {status.value}")

# unique 会防止重复值
try:
    class Duplicate(Enum):
        A = 1
        B = 1
except ValueError as e:
    print("Duplicate error:", e)
