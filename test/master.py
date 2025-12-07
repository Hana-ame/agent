import os
import subprocess

result = subprocess.run("py slave.py 1 2", shell=True, capture_output=True, text=True)

# 获取标准输出
print("输出内容:", result.stdout)

# 获取错误输出（如果有）
print("错误内容:", result.stderr)

# 获取退出码
print("退出码:", result.returncode)

print(result)