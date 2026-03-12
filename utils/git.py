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
    processed_args = []
    for arg in args:
        # 1. 修复 shlex(posix=False) 可能残留的外层引号
        # 当 CommandExecutor 使用 posix=False 时，引号会保留在字符串内，我们需要手动剥离
        if len(arg) >= 2 and (
            (arg.startswith('"') and arg.endswith('"')) or 
            (arg.startswith("'") and arg.endswith("'"))
        ):
            arg = arg[1:-1]
        
        # 2. 处理换行符：允许在 commit message 等地方使用 \n
        arg = arg.replace("\\n", "\n")
        processed_args.append(arg)
    
    cmd = ["git"] + processed_args
    
    try:
        # 在 ctx.root_path 下执行
        result = subprocess.run(
            cmd,
            cwd=ctx.root_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # 如果 git 命令执行失败（返回码非0）
        if result.returncode != 0:
            # 提取 stderr 内容，如果 stderr 为空则显示返回码
            error_msg = result.stderr.strip() or f"Git 命令返回非零状态码 {result.returncode}"
            return f"=== git ===\n错误：{error_msg}\n=== end of git ==="
        
        # 成功执行
        stdout_output = result.stdout.strip()
        stderr_output = result.stderr.strip()
        
        # 合并输出
        final_output = stdout_output
        if stderr_output:
            # Git 有些正常指令也会输出到 stderr（如 checkout, clone 的进度）
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