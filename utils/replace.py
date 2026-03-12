"""
replace - 替换文件中的文本

用法：py utils.py replace <相对路径> <旧文本> <新文本>

参数：
  <相对路径>  要操作的文件路径
  <旧文本>    被替换的字符串
  <新文本>    替换后的字符串

注意：执行的是简单字符串替换，不支持正则表达式。

成功输出格式：成功：已将 <相对路径> 中的 '<旧文本>' 替换为 '<新文本>'
失败输出格式：
  === <相对路径> ===
  错误：具体错误信息
  === end of <相对路径> ===
"""

def run(ctx, args):
    if len(args) != 3:
        return "错误：replace 需要 3 个参数：<相对路径> <旧文本> <新文本>"
    rel_path = args[0]
    old, new = args[1], args[2]
    try:
        full_path = ctx.validate_path(rel_path)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        new_content = content.replace(old, new)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"成功：已将 {rel_path} 中的 '{old}' 替换为 '{new}'"
    except Exception as e:
        return f"=== {rel_path} ===\n错误：{str(e)}\n=== end of {rel_path} ==="