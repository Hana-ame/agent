"""
pause - 暂停 Agent 执行，等待人工干预

用法：py utils.py pause

在根目录创建 .pause 文件，Agent 检测到该文件后会暂停循环，
直到执行 resume 或手动删除 .pause 文件。

如非必要，请不要依靠人工干预。
"""

import os

def run(ctx, args):
    pause_file = os.path.join(ctx.root_path, ".pause")
    try:
        with open(pause_file, "w") as f:
            f.write("paused")
        return "请再次确认：1.是否已经写入文件。2.是否已经commit。3.是否已经完成全部任务。4.是否已经将修改记录保存下来。如果有任意一项没有完成，请调用py utils.py resume继续完成"
    except Exception as e:
        return f"错误：无法设置暂停标志 - {e}"
