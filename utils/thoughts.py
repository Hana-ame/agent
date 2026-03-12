"""
thoughts - 管理想法，每个想法存储为 .thoughts/ 目录下的独立文件（文件名包含微秒时间戳）

用法：
    py utils.py thoughts peek         # 查看第一条想法（最早创建）
    py utils.py thoughts pop          # 弹出第一条想法（删除并返回）
    py utils.py thoughts list         # 列出所有想法（按时间顺序，带编号）
    py utils.py thoughts add <想法>   # 添加一条新想法
    py utils.py thoughts clear        # 清空所有想法

输出格式：
    成功时：返回相应的文本。
    失败时：
      === thoughts <子命令> ===
      错误：具体错误信息
      === end of thoughts <子命令> ===
"""

import os
from datetime import datetime
from pathlib import Path

def _get_thoughts_dir(root_path):
    """获取 .thoughts 目录的路径"""
    return Path(root_path) / ".thoughts"

def _ensure_dir(path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)

def _get_timestamp():
    """返回包含微秒的时间戳，确保文件名按时间顺序正确"""
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")

def _list_thought_files(root_path):
    """返回按文件名排序的想法文件列表（不含路径）"""
    thoughts_dir = _get_thoughts_dir(root_path)
    if not thoughts_dir.exists():
        return []
    files = [f for f in thoughts_dir.iterdir() if f.is_file()]
    files.sort(key=lambda x: x.name)  # 升序，最早的文件在前
    return files

def pop_thought(root_path):
    """
    弹出第一条想法（删除并返回），如果没有则返回 None。
    供 agent 内部调用。
    """
    files = _list_thought_files(root_path)
    if not files:
        return None
    first = files[0]
    try:
        with open(first, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        first.unlink()
        return content
    except Exception:
        return None

def list_thoughts(root_path):
    """返回所有想法内容列表（按时间顺序）"""
    files = _list_thought_files(root_path)
    thoughts = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                thoughts.append(fh.read().strip())
        except Exception:
            thoughts.append("(读取失败)")
    return thoughts

def add_thought(root_path, thought):
    """添加一条想法，生成带微秒时间戳的文件名，确保唯一且有序"""
    thoughts_dir = _get_thoughts_dir(root_path)
    _ensure_dir(thoughts_dir)
    timestamp = _get_timestamp()
    filename = timestamp + ".txt"
    filepath = thoughts_dir / filename
    # 极低概率冲突（同一微秒），如果存在则附加序号，但通常不会
    counter = 1
    while filepath.exists():
        filename = f"{timestamp}-{counter}.txt"
        filepath = thoughts_dir / filename
        counter += 1
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(thought)
        return True
    except Exception:
        return False

def clear_thoughts(root_path):
    """清空 .thoughts 目录"""
    thoughts_dir = _get_thoughts_dir(root_path)
    try:
        if thoughts_dir.exists():
            for f in thoughts_dir.iterdir():
                if f.is_file():
                    f.unlink()
            thoughts_dir.rmdir()  # 删除空目录（可选）
        return True
    except Exception:
        return False

def _handle_error(subcmd, msg):
    return f"=== thoughts {subcmd} ===\n错误：{msg}\n=== end of thoughts {subcmd} ==="

def run(ctx, args):
    if not args:
        return __doc__.strip()
    subcmd = args[0].lower()
    root_path = ctx.root_path

    if subcmd == "pop":
        thought = pop_thought(root_path)
        if thought is None:
            return _handle_error(subcmd, "没有更多想法。")
        return thought

    elif subcmd == "peek":
        files = _list_thought_files(root_path)
        if not files:
            return _handle_error(subcmd, "没有想法。")
        try:
            with open(files[0], 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            return _handle_error(subcmd, f"读取失败: {e}")

    elif subcmd == "list":
        thoughts = list_thoughts(root_path)
        if not thoughts:
            return "暂无想法。"
        lines = [f"{i+1}. {t}" for i, t in enumerate(thoughts)]
        return "\n".join(lines)

    elif subcmd == "add":
        if len(args) < 2:
            return _handle_error(subcmd, "缺少想法内容。")
        thought = " ".join(args[1:])
        if add_thought(root_path, thought):
            return f"已添加想法：{thought}"
        else:
            return _handle_error(subcmd, "添加失败。")

    elif subcmd == "clear":
        if clear_thoughts(root_path):
            return "所有想法已清空。"
        else:
            return _handle_error(subcmd, "清空失败。")

    else:
        return _handle_error(subcmd, f"未知的子命令: {subcmd}")