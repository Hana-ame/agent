import sys

if sys.version_info >= (3, 11):
    import tomllib
    data = """
    [project]
    name = "myapp"
    version = "1.0"
    """
    parsed = tomllib.loads(data)
    print(parsed)
else:
    print("tomllib requires Python 3.11+")
