import sys
import importlib

packages = [
    ('requests', 'requests'),
    ('beautifulsoup4', 'bs4'),
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('matplotlib', 'matplotlib'),
    ('flask', 'flask'),
    ('pytest', 'pytest'),
    ('sqlalchemy', 'sqlalchemy'),
    ('aiohttp', 'aiohttp'),
]

print("Python version:", sys.version)
for pkg_name, import_name in packages:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"{pkg_name:15} {version}")
    except ImportError:
        print(f"{pkg_name:15} NOT INSTALLED")
