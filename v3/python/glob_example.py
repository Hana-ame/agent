import glob
import fnmatch
import os

# 创建测试文件
os.makedirs("glob_test", exist_ok=True)
for f in ["file1.txt", "file2.txt", "file3.py", "data.csv"]:
    open(f"glob_test/{f}", 'w').close()

# glob
print("*.txt:", glob.glob("glob_test/*.txt"))
print("*.py:", glob.glob("glob_test/*.py"))

# fnmatch 过滤
names = os.listdir("glob_test")
print("Matches *.txt:", [n for n in names if fnmatch.fnmatch(n, "*.txt")])

# 清理
import shutil
shutil.rmtree("glob_test")
