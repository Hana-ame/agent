import datetime
import time as time_module
import os
import argparse
from pathlib import Path

def run(ctx, args):
    """
    时间工具：显示当前日期时间，或管理计时器。
    
    用法:
        py utils.py time                    显示当前日期和时间
        py utils.py time --date              只显示当前日期
        py utils.py time --time              只显示当前时间
        py utils.py time --timestamp         显示 Unix 时间戳
        py utils.py time --start             开始计时（保存开始时间）
        py utils.py time --stop               停止计时并显示经过时间
        py utils.py time --elapsed            显示当前经过时间（不停止）
        py utils.py time --format <格式>      使用自定义格式显示当前时间
                                             格式字符串遵循 strftime 规则
    """
    parser = argparse.ArgumentParser(description="时间与计时工具", add_help=False)
    parser.add_argument("--date", action="store_true", help="只显示当前日期")
    parser.add_argument("--time", action="store_true", help="只显示当前时间")
    parser.add_argument("--timestamp", action="store_true", help="显示当前 Unix 时间戳")
    parser.add_argument("--start", action="store_true", help="开始计时")
    parser.add_argument("--stop", action="store_true", help="停止计时并显示经过时间")
    parser.add_argument("--elapsed", action="store_true", help="显示当前经过时间")
    parser.add_argument("--format", type=str, help="自定义显示格式（strftime 格式）")
    
    # 解析参数，处理未知参数（避免 argparse 抛出异常）
    try:
        parsed_args = parser.parse_args(args)
    except SystemExit:
        # 如果参数错误，直接显示帮助并返回
        print(parser.format_help())
        return
    
    # 计时器文件路径（存储在项目根目录下）
    timer_file = Path(ctx.root_path) / ".timer_start"
    
    # 如果指定了 --format，优先使用自定义格式显示当前时间
    if parsed_args.format:
        now = datetime.datetime.now()
        try:
            print(now.strftime(parsed_args.format))
        except Exception as e:
            print(f"格式错误: {e}")
        return
    
    # 处理计时相关选项
    if parsed_args.start:
        if timer_file.exists():
            print("计时器已经在运行中。如需重新开始，请先执行 --stop。")
        else:
            timer_file.write_text(str(time_module.time()))
            print("计时器已启动。")
        return
    
    if parsed_args.stop:
        if not timer_file.exists():
            print("没有正在运行的计时器。")
            return
        try:
            start_time = float(timer_file.read_text().strip())
            elapsed = time_module.time() - start_time
            timer_file.unlink()  # 删除文件
            # 格式化显示
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"经过时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        except Exception as e:
            print(f"读取计时器出错: {e}")
        return
    
    if parsed_args.elapsed:
        if not timer_file.exists():
            print("没有正在运行的计时器。")
            return
        try:
            start_time = float(timer_file.read_text().strip())
            elapsed = time_module.time() - start_time
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"当前经过时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        except Exception as e:
            print(f"读取计时器出错: {e}")
        return
    
    # 没有指定任何选项，或只指定了日期/时间选项
    now = datetime.datetime.now()
    if parsed_args.date:
        print(now.strftime("%Y-%m-%d"))
    elif parsed_args.time:
        print(now.strftime("%H:%M:%S"))
    elif parsed_args.timestamp:
        print(int(time_module.time()))
    else:
        # 默认显示完整日期时间
        print(now.strftime("%Y-%m-%d %H:%M:%S"))\n# [START] MAIN\n"\"\"\"\n# [END] MAIN\n