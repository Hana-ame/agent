from pathlib import Path

# 创建项目结构
base = Path("demo_project")
base.mkdir(exist_ok=True)
(base / "src").mkdir(exist_ok=True)
(base / "tests").mkdir(exist_ok=True)
(base / "docs").mkdir(exist_ok=True)

print("Created directories:")
for path in sorted(base.glob("*")):
    print(f"  {path}")

# 清理
import shutil
shutil.rmtree(base)
print("Cleaned up")
