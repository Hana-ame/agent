# [START] TOOL-GIT
# version: 003
# 描述：执行 git 命令

"""
git - 执行 git 命令

用法：py utils.py git <git-subcommand> [args...]

参数：
  任意 git 子命令和参数，透传给 git 执行。

成功时直接返回 git 命令的输出（包含 stdout 和 stderr）。
失败时输出统一格式错误块。
"""

import subprocess

def run(ctx, args):
    if not args:
        return "错误：git 需要子命令 (e.g., status, diff)"
    
    cmd = ["git"] + args
    
    try:
        result = subprocess.run(
            cmd,
            cwd=ctx.root_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # 如果 git 命令执行失败（返回码非0）
        if result.returncode != 0:
            error_msg = result.stderr.strip() or f"Git 命令返回非零状态码 {result.returncode}"
            return f"=== git ===\n错误：{error_msg}\n=== end of git ==="
        
        # 成功，返回输出
        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr
        return output.strip()
    except subprocess.TimeoutExpired:
        return f"=== git ===\n错误：Git 命令超时\n=== end of git ==="
    except Exception as e:
        return f"=== git ===\n错误：Git 执行失败 - {str(e)}\n=== end of git ==="
# [END] TOOL-GIT