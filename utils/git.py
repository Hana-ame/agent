
import subprocess

def run(ctx, args):
    # 如果 git 命令需要操作文件，可使用 ctx.validate_path 确保安全
    # 此处仅作演示，直接执行 git 命令
    try:
        result = subprocess.run(["git"] + args, capture_output=True, text=True, cwd=ctx.root_path)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Git 错误：{e}"