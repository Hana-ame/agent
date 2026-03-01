"""
append - 将指定内容追加到文件末尾

用法：py utils.py append <相对路径> <内容>

参数：
  <相对路径>  相对于根目录的文件路径
  <内容>      要追加的文本内容（多个参数自动用空格连接）

如果文件不存在，会自动创建。
"""

def run(ctx, args):
    if len(args) < 2:
        return "错误：append 需要 2 个参数：<相对路径> <内容>"
    path = ctx.validate_path(args[0])
    content = " ".join(args[1:])
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"成功：已将内容追加到 {args[0]}"
    except Exception as e:
        return f"错误：无法追加到文件 {args[0]} - {e}"
