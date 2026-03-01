import os

def run(ctx, args):
    pause_file = os.path.join(ctx.root_path, ".pause")
    try:
        with open(pause_file, "w") as f:
            f.write("paused")
        return "已设置暂停标志。后续命令将暂停执行，直到运行 'py utils.py resume'。"
    except Exception as e:
        return f"错误：无法设置暂停标志 - {e}"
