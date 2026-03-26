from pathlib import Path
import os

# 创建临时目录结构
temp_dir = Path("temp_structure")
temp_dir.mkdir(exist_ok=True)
(temp_dir / "file1.txt").write_text("content1")
(temp_dir / "file2.py").write_text("print('hello')")
subdir = temp_dir / "subdir"
subdir.mkdir()
(subdir / "file3.txt").write_text("content3")

# glob 匹配
print("Python files:", list(temp_dir.glob("*.py")))
print("All .txt files:", list(temp_dir.glob("**/*.txt")))

# 遍历目录
print("Directory tree:")
for p in temp_dir.rglob("*"):
    print(f"  {p.relative_to(temp_dir)}")

# 清理
import shutil
shutil.rmtree(temp_dir)
