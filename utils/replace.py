# utils/replace.py

def run(ctx, args):
    if len(args) != 3:
        return "错误：replace 需要 3 个参数：<相对路径> <旧文本> <新文本>"
    path = ctx.validate_path(args[0])
    old, new = args[1], args[2]
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        new_content = content.replace(old, new)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"成功：已将 {args[0]} 中的 '{old}' 替换为 '{new}'"
    except Exception as e:
        return f"错误：无法替换文件 {args[0]} 中的内容 - {e}"