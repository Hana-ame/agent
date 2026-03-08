
# [START] TOOL-GIT
# version: 002
# 描述：执行 git 命令
import subprocess

def run(ctx, args):
    if not args:
        return "Error: git 需要子命令 (e.g., status, diff)"
    
    # 限制只能运行 git 命令
    cmd = ["git"] + args
    
    try:
        # 在 ctx.root_path 下执行
        result = subprocess.run(
            cmd,
            cwd=ctx.root_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr
            
        return output.strip()
    except subprocess.TimeoutExpired:
        return "Error: Git 命令超时"
    except Exception as e:
        return f"Error: Git 执行失败 - {str(e)}"
# [END] TOOL-GIT