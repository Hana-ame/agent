import subprocess

# 运行命令并捕获输出
result = subprocess.run(['echo', 'Hello from subprocess'], capture_output=True, text=True)
print("stdout:", result.stdout.strip())

# 检查命令是否存在
try:
    subprocess.run(['python', '--version'], check=True, capture_output=True, text=True)
    print("Python is available")
except subprocess.CalledProcessError:
    print("Python command failed")
