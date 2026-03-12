# [START] TOOL-CP
# version: 0.0.1
# 描述：复制文件或目录

"""
cp - 复制文件或目录

用法：
    py utils.py cp [-r] <源路径> <目标路径>

选项：
    -r    递归复制目录

说明：
    - 如果目标是已存在的目录，源会被复制到该目录内，保留原名。
    - 如果目标是一个文件（或不存在且不以 / 结尾），则执行复制/重命名。
    - 支持递归复制目录（需要 -r 选项）。
    - 目标路径的父目录必须存在（对于文件复制）或自动创建（对于目录复制，但需要 -r）。
    - 源路径必须存在，否则报错。

成功输出格式：成功：已将 '<源>' 复制到 '<目标>'
失败输出格式：
    === cp ===
    错误：具体错误信息
    === end of cp ===
"""

import os
import shutil

def _handle_error(subcmd: str, msg: str) -> str:
    return f"""=== {subcmd} ===
错误：{msg}
=== end of {subcmd} ==="""

def run(ctx, args):
    if len(args) < 2:
        return _handle_error("cp", "至少需要源路径和目标路径两个参数。用法：cp [-r] <源> <目标>")
    
    # 解析 -r 选项
    recursive = False
    src_raw = None
    dst_raw = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '-r':
            recursive = True
            i += 1
        else:
            if src_raw is None:
                src_raw = arg
            elif dst_raw is None:
                dst_raw = arg
            else:
                return _handle_error("cp", "参数过多。")
            i += 1
    
    if src_raw is None or dst_raw is None:
        return _handle_error("cp", "必须提供源路径和目标路径。")

    # 验证路径
    try:
        src_path = ctx.validate_path(src_raw)
    except Exception as e:
        return _handle_error("cp", f"无效的源路径 '{src_raw}': {e}")

    try:
        dst_path = ctx.validate_path(dst_raw)
    except Exception as e:
        return _handle_error("cp", f"无效的目标路径 '{dst_raw}': {e}")

    # 检查源是否存在
    if not os.path.exists(src_path):
        return _handle_error("cp", f"源路径不存在 '{src_raw}'")

    # 判断目标是否为目录（如果目标已存在且是目录，则源将被复制到该目录下）
    if os.path.isdir(dst_path):
        # 目标是一个已存在的目录，源将被复制到该目录内，保留原名
        dst_path = os.path.join(dst_path, os.path.basename(src_path))
    elif dst_raw.endswith('/') or dst_raw.endswith('\\'):
        # 目标以路径分隔符结尾，视为目录（即使不存在），需要创建
        if not os.path.exists(dst_path):
            try:
                os.makedirs(dst_path, exist_ok=True)
            except Exception as e:
                return _handle_error("cp", f"无法创建目标目录 '{dst_raw}': {e}")
        dst_path = os.path.join(dst_path, os.path.basename(src_path))

    # 确保目标父目录存在
    dst_parent = os.path.dirname(dst_path)
    if dst_parent and not os.path.exists(dst_parent):
        try:
            os.makedirs(dst_parent, exist_ok=True)
        except Exception as e:
            return _handle_error("cp", f"无法创建目标父目录 '{dst_parent}': {e}")

    try:
        if os.path.isdir(src_path):
            if not recursive:
                return _handle_error("cp", f"源 '{src_raw}' 是一个目录，需要使用 -r 选项进行递归复制。")
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
        # 计算相对路径用于输出
        rel_dst = os.path.relpath(dst_path, ctx.root_path)
        return f"成功：已将 '{src_raw}' 复制到 '{rel_dst}'"
    except Exception as e:
        return _handle_error("cp", f"复制失败 - {e}")


# [END] TOOL-CP