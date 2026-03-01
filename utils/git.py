"""
git - 执行 Git 命令

用法：py utils.py git <git子命令> [参数...]

示例：
  py utils.py git status
  py utils.py git add .
  py utils.py git commit -m "message"

所有命令在根目录下执行。
"""

import subprocess


def run(ctx, args):
    try:
        result = subprocess.run(
            ["git"] + args, capture_output=True, text=True, cwd=ctx.root_path
        )
        return "git 返回了下面的信息：、\n" + result.stdout + result.stderr
    except Exception as e:
        return f"Git 错误：{e}"
