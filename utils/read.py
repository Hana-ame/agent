"""
read - 读取文件内容

用法：py utils.py read <相对路径> [<相对路径2> ...]

参数：
  <相对路径>  要读取的文件路径（相对于根目录），可指定多个

如果只指定一个文件，直接返回文件内容（不带标记）。
如果指定多个文件，每个文件内容前会加上 "=== 文件名 ===" 标记以便区分。
"""

def run(ctx, args):
    if len(args) == 0:
        return "Error: read requires at least one argument: <relative_path> [<relative_path2> ...]"
    
    outputs = []
    
    if len(args) == 1:
        # 单个文件：直接返回内容，保持向后兼容
        path = ctx.validate_path(args[0])
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error: cannot read file {args[0]} - {e}"
    else:
        # 多个文件：输出带标记的内容
        for file_arg in args:
            path = ctx.validate_path(file_arg)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                outputs.append(f"=== {file_arg} ===\n{content}\n===\n")
            except Exception as e:
                outputs.append(f"=== {file_arg} ===\nError: cannot read file - {e}\n")
        return "\n\n".join(outputs)