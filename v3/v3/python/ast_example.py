import ast
import inspect

code = """
def add(a, b):
    return a + b
"""

tree = ast.parse(code)
print("AST dump:")
print(ast.dump(tree, indent=2))

# 执行 AST 编译的代码
module = compile(tree, '<string>', 'exec')
namespace = {}
exec(module, namespace)
print("Result:", namespace['add'](3, 5))
