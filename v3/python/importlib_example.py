import importlib
import sys

# 动态导入模块
math = importlib.import_module('math')
print(f"math.sqrt(16) = {math.sqrt(16)}")

# 重新加载模块（如果修改了）
if 'json' in sys.modules:
    importlib.reload(sys.modules['json'])
    print("json module reloaded")
