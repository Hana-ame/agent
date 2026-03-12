"""
time - 显示当前时间、日期、时间戳，或管理计时器

用法：
    py utils.py time                    显示当前日期和时间
    py utils.py time --date              只显示当前日期
    py utils.py time --time              只显示当前时间
    py utils.py time --timestamp         显示 Unix 时间戳
    py utils.py time --start             开始计时（保存开始时间）
    py utils.py time --stop               停止计时并显示经过时间
    py utils.py time --elapsed            显示当前经过时间（不停止）
    py utils.py time --format <格式>      使用自定义格式显示当前时间（strftime 格式）

输出格式：
    成功时：返回相应的文本。
    失败时：
      === time <子命令> ===
      错误：具体错误信息
      === end of time <子命令> ===
"""

import datetime
import time as time_module
import os
import argparse
from pathlib import Path

def _handle_error(subcmd: str, msg: str) -> str:
    return f"=== time {subcmd} ===\n错误：{msg}\n=== end of time {subcmd} ==="

def run(ctx, args):
    parser = argparse.ArgumentParser(description="时间与计时工具", add_help=False)
    parser.add_argument("--date", action="store_true", help="只显示当前日期")
    parser.add_argument("--time", action="store_true", help="只显示当前时间")
    parser.add_argument("--timestamp", action="store_true", help="显示当前 Unix 时间戳")
    parser.add_argument("--start", action="store_true", help="开始计时")
    parser.add_argument("--stop", action="store_true", help="停止计时并显示经过时间")
    parser.add_argument("--elapsed", action="store_true", help="显示当前经过时间")
    parser.add_argument("--format", type=str, help="自定义显示格式（strftime 格式）")

    # 如果没有参数，默认显示完整日期时间
    if not args:
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit:
        # 参数解析失败（如提供了未知选项）
        return _handle_error("", "参数错误，请检查用法。\n" + __doc__.strip())

    # 计时器文件路径
    timer_file = Path(ctx.root_path) / ".timer_start"

    # 如果指定了 --format，优先使用自定义格式显示当前时间
    if parsed_args.format:
        now = datetime.datetime.now()
        try:
            return now.strftime(parsed_args.format)
        except Exception as e:
            return _handle_error("format", f"格式错误: {e}")

    # 处理计时相关选项
    if parsed_args.start:
        if timer_file.exists():
            return _handle_error("start", "计时器已经在运行中。如需重新开始，请先执行 --stop。")
        timer_file.write_text(str(time_module.time()))
        return "计时器已启动。"

    if parsed_args.stop:
        if not timer_file.exists():
            return _handle_error("stop", "没有正在运行的计时器。")
        try:
            start_time = float(timer_file.read_text().strip())
            elapsed = time_module.time() - start_time
            timer_file.unlink()  # 删除文件
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            return f"经过时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
        except Exception as e:
            return _handle_error("stop", f"读取计时器出错: {e}")

    if parsed_args.elapsed:
        if not timer_file.exists():
            return _handle_error("elapsed", "没有正在运行的计时器。")
        try:
            start_time = float(timer_file.read_text().strip())
            elapsed = time_module.time() - start_time
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            return f"当前经过时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
        except Exception as e:
            return _handle_error("elapsed", f"读取计时器出错: {e}")

    # 没有指定计时选项，处理日期/时间/时间戳
    now = datetime.datetime.now()
    if parsed_args.date:
        return now.strftime("%Y-%m-%d")
    elif parsed_args.time:
        return now.strftime("%H:%M:%S")
    elif parsed_args.timestamp:
        return str(int(time_module.time()))
    else:
        # 默认（理论上不会走到这里，因为前面已经处理了无参数的情况）
        return now.strftime("%Y-%m-%d %H:%M:%S")