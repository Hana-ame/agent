# -*- coding: utf-8 -*-

"""
mv - 移动或重命名文件/目录

用法：py utils.py mv <源路径> <目标路径>

参数：
  <源路径>     要移动的文件或目录的相对路径
  <目标路径>   目标位置的相对路径

说明：
  - 如果目标路径是一个已存在的目录，则源会被移动到该目录内，保留原名。
  - 如果目标路径是一个文件（或不存在但包含文件名），则执行重命名/移动。
  - 支持跨文件系统移动（使用 shutil.move）。
  - 如果目标文件已存在，将会被覆盖（需要适当权限）。
  - 源路径必须存在，否则报错。
  - 目标路径的父目录必须存在，否则报错。

成功输出格式：成功：已将 '<源路径>' 移动到 '<目标路径>'
失败输出格式：
  === <源路径> ===
  错误：具体错误信息
  === end of <源路径> ===
"""

import os
import shutil

def run(ctx, args):
    if len(args) != 2:
        return "错误：mv 需要 2 个参数：<源路径> <目标路径>"

    src_raw, dst_raw = args[0], args[1]

    # 获取绝对路径并验证在根目录内
    try:
        src_path = ctx.validate_path(src_raw)
    except Exception as e:
        return f"=== {src_raw} ===\n错误：无效的源路径 - {e}\n=== end of {src_raw} ==="

    try:
        dst_path = ctx.validate_path(dst_raw)
    except Exception as e:
        return f"=== {src_raw} ===\n错误：无效的目标路径 - {e}\n=== end of {src_raw} ==="

    # 检查源是否存在
    if not os.path.exists(src_path):
        return f"=== {src_raw} ===\n错误：源路径不存在\n=== end of {src_raw} ==="

    # 检查目标父目录是否存在
    dst_parent = os.path.dirname(dst_path)
    if dst_parent and not os.path.exists(dst_parent):
        return f"=== {src_raw} ===\n错误：目标路径的父目录不存在 '{os.path.dirname(dst_raw)}'\n=== end of {src_raw} ==="

    # 如果目标已存在且是目录，则将源移动到该目录下（保留原名）
    if os.path.isdir(dst_path):
        dst_path = os.path.join(dst_path, os.path.basename(src_path))

    try:
        shutil.move(src_path, dst_path)
        # 计算相对于根目录的目标路径用于输出
        rel_dst = os.path.relpath(dst_path, ctx.root_path)
        return f"成功：已将 '{src_raw}' 移动到 '{rel_dst}'"
    except Exception as e:
        return f"=== {src_raw} ===\n错误：移动失败 - {e}\n=== end of {src_raw} ==="