# [START] TOOL-WRITE
# version: 0.0.3
# 描述：覆盖写入文件

"""
write - 覆盖写入文件

用法：py utils.py write <相对路径> <内容...>

参数：
  <相对路径>  相对于根目录的文件路径
  <内容...>   要写入的文本内容（多个参数自动用空格连接）

说明：
  - 自动创建父目录。
  - 可以使用 `\\n` 表示换行符（会被转换为真实换行）。
  - 内容较长时会自动截断预览。

输出格式：
  - 成功时：返回一行：
    {rel_path}中被写入了以下内容
    {内容预览}
  - 失败时：
    * 参数不足返回：
      === write ===
      错误：write 需要至少 2 个参数：<路径> <内容...>
      === end of write ===
    * 写入失败返回针对该文件的错误块：
      === {rel_path} ===
      错误：具体错误信息
      === end of {rel_path} ===
"""
import subprocess

def run(ctx, args):
    if not args:
        return "错误：git 需要子命令 (e.g., status, diff)"
    
    # --- 参数预处理 ---
    processed_args =[]
    for arg in args:
        # 1. 不再需要手动剥离引号，因为 executor 已经以 posix 模式剥离干净
        # 2. 只需要处理换行符：将字面量 \n 转化为真实的换行
        arg = arg.replace("\\n", "\n")
        processed_args.append(arg)
    
    cmd = ["git"] + processed_args
    
    try:
        result = subprocess.run(
            cmd,
            cwd=ctx.root_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            error_msg = result.stderr.strip() or f"Git 命令返回非零状态码 {result.returncode}"
            return f"=== git ===\n错误：{error_msg}\n=== end of git ==="
        
        stdout_output = result.stdout.strip()
        stderr_output = result.stderr.strip()
        final_output = stdout_output
        if stderr_output:
            if final_output:
                final_output += "\n" + stderr_output
            else:
                final_output = stderr_output
                
        return final_output if final_output else "执行成功 (无输出)"

    except subprocess.TimeoutExpired:
        return "=== git ===\n错误：Git 命令超时\n=== end of git ==="
    except Exception as e:
        return f"=== git ===\n错误：Git 执行失败 - {str(e)}\n=== end of git ==="
# [END] TOOL-GIT