import ast
import os
import sys

def get_imports(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=filepath)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def main():
    main_file = 'agent_v3.py'
    if not os.path.exists(main_file):
        print(f"错误：{main_file} 不存在", file=sys.stderr)
        sys.exit(1)
    
    imports = get_imports(main_file)
    deps = {main_file}
    for mod in imports:
        candidate = f"{mod}.py"
        if os.path.exists(candidate):
            deps.add(candidate)
        elif os.path.isdir(mod) and os.path.exists(os.path.join(mod, "__init__.py")):
            deps.add(os.path.join(mod, "__init__.py"))
    print("\n".join(deps))

if __name__ == '__main__':
    main()