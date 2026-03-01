=== utils/delete.py ===
"""
delete - 删除一个或多个文件

用法：py utils.py delete <相对路径1> [<相对路径2> ...]

参数：
  一个或多个要删除的文件路径（相对于根目录）

注意：
  - 只能删除文件，不能删除目录。
  - 如果提供了多个路径，将依次删除每个文件。
  - 删除失败不会中断后续文件的删除操作。
  - 执行完成后会报告成功和失败的数量。
"""

import os

def run(ctx, args):
    """
    执行删除操作。
    args: 一个或多个文件路径的列表。
    """
    if len(args) == 0:
        return "错误：delete 需要至少一个参数：<相对路径1> [<相对路径2> ...]"

    success_count = 0
    fail_count = 0
    fail_messages = []

    for path_str in args:
        try:
            path = ctx.validate_path(path_str)
            os.remove(path)
            success_count += 1
        except Exception as e:
            fail_count += 1
            fail_messages.append(f"  {path_str}: {e}")

    # 构建结果消息
    result = f"删除完成：成功 {success_count}，失败 {fail_count}"
    if fail_messages:
        result += "\n失败详情：\n" + "\n".join(fail_messages)
    return result
===