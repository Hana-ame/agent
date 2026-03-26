import zipfile
import tempfile
import sys

# 创建一个 zip 文件包含一个模块
with tempfile.NamedTemporaryFile(suffix='.zip') as tmp:
    with zipfile.ZipFile(tmp.name, 'w') as zf:
        zf.writestr('mymodule.py', 'def greet(): return "Hello from zip"\n')
    
    # 添加 zip 到路径并导入
    sys.path.insert(0, tmp.name)
    import mymodule
    print(mymodule.greet())
    sys.path.remove(tmp.name)
