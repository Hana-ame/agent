from pathlib import Path

# 创建临时目录和文件
tmp_dir = Path("temp_test")
tmp_dir.mkdir(exist_ok=True)

file_path = tmp_dir / "test.txt"
file_path.write_text("Hello, pathlib!")

print("File exists:", file_path.exists())
print("File content:", file_path.read_text())

# 清理
file_path.unlink()
tmp_dir.rmdir()
