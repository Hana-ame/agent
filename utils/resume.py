import os

def run(ctx, args):
    pause_file = os.path.join(ctx.root_path, ".pause")
    if os.path.exists(pause_file):
        try:
            os.remove(pause_file)
            return "已清除暂停标志。可以继续执行命令。"
        except Exception as e:
            return f"错误：无法清除暂停标志 - {e}"
    else:
        return "当前没有暂停标志。"
