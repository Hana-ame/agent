# [START] TOOL-GIT
# version: 0.0.4
# 描述：执行 git 命令

"""
git - 执行 git 命令

用法：py utils.py git <git-subcommand> [args...]

参数：
  任意 git 子命令和参数。支持带空格的参数（需在命令中用引号包裹）以及 \n 换行符。

注意：
  为了处理 commit 信息中的换行，程序会自动将参数中的字面量 "\\n" 转换为实际的换行符。
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