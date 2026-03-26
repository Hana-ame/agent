# 文件处理与异常
import os

# 读取之前创建的文件
try:
    with open("output.txt", "r") as f:
        content = f.read()
    print("Content of output.txt:")
    print(content)
except FileNotFoundError:
    print("File not found")

# 检查文件是否存在并读取行
if os.path.exists("output.txt"):
    with open("output.txt", "r") as f:
        lines = f.readlines()
    print(f"\nNumber of lines: {len(lines)}")
    for i, line in enumerate(lines, 1):
        print(f"Line {i}: {line.rstrip()}")
else:
    print("output.txt does not exist")

# 写入多个文件
with open("greeting.txt", "w") as f:
    f.write("Hello from Python exercise!\n")
with open("greeting.txt", "a") as f:
    f.write("Appended line.\n")
print("\n'greeting.txt' created with two lines.")
