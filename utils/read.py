"""
read - 读取文件内容

用法：py utils.py read <相对路径>

参数：
  <相对路径>  要读取的文件路径（相对于根目录）

返回文件内容。为兼容控制台编码，非 ASCII 字符会被替换为 '?'。
"""

def run(ctx, args):
    if len(args) != 1:
        return "Error: read requires 1 argument: <relative_path>"
    path = ctx.validate_path(args[0])
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        # Replace non-ASCII characters with '?' to avoid encoding errors when printing
        content = content.encode('GBK', errors='replace').decode('GBK')
        return content
    except Exception as e:
        return f"Error: cannot read file {args[0]} - {e}"
