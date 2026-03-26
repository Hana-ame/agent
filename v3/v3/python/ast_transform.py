import ast
import astor  # 如果未安装，使用内置的 unparse (Python 3.9+)

code = """
def add(a, b):
    return a + b
"""

tree = ast.parse(code)

# 将所有加法改为乘法
class MultiplyTransformer(ast.NodeTransformer):
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            return ast.BinOp(left=node.left, op=ast.Mult(), right=node.right)
        return node

new_tree = MultiplyTransformer().visit(tree)
# 尝试使用 ast.unparse (Python 3.9+)
if hasattr(ast, 'unparse'):
    print(ast.unparse(new_tree))
else:
    # fallback: 使用 astor 如果安装
    try:
        import astor
        print(astor.to_source(new_tree))
    except ImportError:
        print("ast.unparse requires Python 3.9+ or astor installed")
