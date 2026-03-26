# 基础语法练习
print("=== Python Basic Syntax Exercise ===\n")

# 1. 变量与数据类型
name = "Python Learner"
age = 25
height = 1.75
is_student = True

print(f"Name: {name} (type: {type(name).__name__})")
print(f"Age: {age} (type: {type(age).__name__})")
print(f"Height: {height} (type: {type(height).__name__})")
print(f"Is student: {is_student} (type: {type(is_student).__name__})\n")

# 2. 列表与字典
fruits = ["apple", "banana", "orange"]
person = {"name": "Alice", "age": 30}

print(f"Fruits: {fruits}")
print(f"First fruit: {fruits[0]}")
print(f"Person: {person}")
print(f"Person's name: {person['name']}\n")

# 3. 条件语句
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"
print(f"Score: {score}, Grade: {grade}\n")

# 4. 循环
print("Loop through fruits:")
for fruit in fruits:
    print(f"  - {fruit}")

print("\nCountdown:")
for i in range(5, 0, -1):
    print(f"  {i}")
print("  Blast off!\n")

# 5. 函数
def add(a, b):
    return a + b

result = add(10, 20)
print(f"add(10, 20) = {result}\n")

# 6. 文件写入
with open("output.txt", "w") as f:
    f.write("This is a test file.\nLine 2\nLine 3")

print("File 'output.txt' created successfully.")
