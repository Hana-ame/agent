#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
move - 移动或重命名文件/目录

用法：py utils.py move <源路径> <目标路径>

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

示例：
  py utils.py move file.txt newname.txt          # 重命名
  py utils.py move file.txt dir/                  # 移动到 dir/ 目录下
  py utils.py move dir1/ dir2/                    # 移动目录
"""

import os
import shutil
from pathlib import Path

def run(ctx, args):
    if len(args) != 2:
        return "错误：move 需要 2 个参数：<源路径> <目标路径>"

    src_raw, dst_raw = args[0], args[1]

    # 获取绝对路径并验证在根目录内
    try:
        src_path = ctx.validate_path(src_raw)
    except Exception as e:
        return f"错误：无效的源路径 '{src_raw}': {e}"

    try:
        dst_path = ctx.validate_path(dst_raw)
    except Exception as e:
        return f"错误：无效的目标路径 '{dst_raw}': {e}"

    # 检查源是否存在
    if not os.path.exists(src_path):
        return f"错误：源路径不存在 '{src_raw}'"

    # 检查目标父目录是否存在
    dst_parent = os.path.dirname(dst_path)
    if dst_parent and not os.path.exists(dst_parent):
        return f"错误：目标路径的父目录不存在 '{os.path.dirname(dst_raw)}'"

    # 如果目标已存在且是目录，则将源移动到该目录下（保留原名）
    if os.path.isdir(dst_path):
        dst_path = os.path.join(dst_path, os.path.basename(src_path))

    try:
        shutil.move(src_path, dst_path)
        return f"成功：已将 '{src_raw}' 移动到 '{os.path.relpath(dst_path, ctx.root_path)}'"
    except Exception as e:
        return f"错误：移动失败 - {e}"