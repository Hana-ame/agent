"""
thoughts - 管理 THOUGHTS.md 中的想法

用法：
    py utils.py thoughts pop          # 弹出第一条想法并返回
    py utils.py thoughts peek         # 查看第一条想法但不删除
    py utils.py thoughts list         # 列出所有想法（带编号）
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
from pathlib import Path

def _get_thoughts_file(root_path):
    """获取 THOUGHTS.md 的路径"""
    return Path(root_path) / ".agent" / "THOUGHTS.md"

def pop_thought(root_path):
    """
    弹出第一条想法（删除并返回），如果没有则返回 None。
    供 agent 内部调用。
    """
    thoughts_file = _get_thoughts_file(root_path)
    if not thoughts_file.exists():
        return None
    try:
        with open(thoughts_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        new_lines = []
        thought = None
        for line in lines:
            stripped = line.lstrip()
            if thought is None and stripped.startswith('- '):
                thought = stripped[2:].strip()
                # 跳过该行
            else:
                new_lines.append(line)
        if thought:
            with open(thoughts_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            return thought
        else:
            return None
    except Exception:
        return None

def list_thoughts(root_path):
    """列出所有想法（返回列表）"""
    thoughts_file = _get_thoughts_file(root_path)
    if not thoughts_file.exists():
        return []
    try:
        with open(thoughts_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        thoughts = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('- '):
                thoughts.append(stripped[2:].strip())
        return thoughts
    except Exception:
        return []

def add_thought(root_path, thought):
    """添加一条想法"""
    thoughts_file = _get_thoughts_file(root_path)
    thoughts_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(thoughts_file, 'a', encoding='utf-8') as f:
            f.write(f"- {thought}\n")
        return True
    except Exception:
        return False

def clear_thoughts(root_path):
    """清空想法文件"""
    thoughts_file = _get_thoughts_file(root_path)
    try:
        if thoughts_file.exists():
            thoughts_file.unlink()
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
        thoughts = list_thoughts(root_path)
        if not thoughts:
            return _handle_error(subcmd, "没有想法。")
        return thoughts[0]

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