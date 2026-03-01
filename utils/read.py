
# utils/read.py

def run(ctx, args):
    if len(args) != 1:
        return "错误：read 需要 1 个参数：<相对路径>"
    path = ctx.validate_path(args[0])
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"错误：无法读取文件 {args[0]} - {e}"