import shlex, sys, subprocess, os

command = "py utils.py read main.py"
# 分割命令，并将 'py' 替换为当前 Python 解释器路径
parts = shlex.split(command)
if parts and parts[0] in ('py', 'python', 'python3'):
  parts[0] = sys.executable
result = subprocess.run(parts, capture_output=True, text=True, cwd=os.getcwd())
output = result.stdout + result.stderr
if result.returncode != 0:
  output = f"命令执行失败 (返回码 {result.returncode}):\n{output}"
else:
  output = output.strip()

print(output)