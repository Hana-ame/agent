#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
delete - 删除一个或多个文件

用法：py utils.py delete <相对路径1> [<相对路径2> ...]

参数：
  一个或多个要删除的文件路径（相对于根目录）

注意：
  - 只能删除文件，不能删除目录。
  - 如果提供了多个路径，将依次删除每个文件。
  - 删除失败不会中断后续文件的删除操作。
  - 执行完成后会报告成功和失败的数量，失败时输出统一格式的错误块。
"""

import os

def run(ctx, args):
    if len(args) == 0:
        return "错误：delete 需要至少一个参数：<相对路径1> [<相对路径2> ...]"
    
    success_count = 0
    error_blocks = []
    
    for rel_path in args:
        try:
            full_path = ctx.validate_path(rel_path)
            os.remove(full_path)
            success_count += 1
        except Exception as e:
            error_blocks.append(f"=== {rel_path} ===\n错误：{e}\n=== end of {rel_path} ===")
    
    result = f"删除完成：成功 {success_count}，失败 {len(error_blocks)}"
    if error_blocks:
        result += "\n\n" + "\n\n".join(error_blocks)
    return result