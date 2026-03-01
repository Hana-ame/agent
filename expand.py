# -*- coding: utf-8 -*-

"""
文件处理逻辑（本地版）：
1. 正则匹配：使用正则表达式 `\[([^\]]+)\]` 捕获方括号内的相对路径。
2. 路径拼接：将相对路径与指定的基础目录（绝对路径）拼接，得到完整文件路径。
3. 文件读取：尝试以 UTF-8 编码读取文件内容。
4. 内容插入：将获取到的文本内容进行格式化，紧随在原路径的下一行，并添加分割线以示区别。
5. 错误处理：如果文件不存在或无法读取，则在原位置注明错误信息，不中断程序。
6. 编码处理：统一使用 UTF-8 编码读取和写入，防止中文乱码。

基础目录获取优先级（高 -> 低）：
   - 命令行参数 --base-dir / -b
   - 当前目录下的 .env 文件中的 BASE_DIR 变量
   - 当前目录下的 config.txt 文件的第一行（自动去除首尾空白）
   - 当前工作目录（os.getcwd()）

使用方法：
    python expand.py <输入文件> <输出文件> [--base-dir <路径>]

示例：
    python expand.py README.md README_expanded.md --base-dir /home/user/docs
"""

import re
import sys
import os
import argparse
from pathlib import Path

def get_base_dir(args):
    """根据优先级获取基础目录"""
    # 1. 命令行参数
    if args.base_dir:
        return os.path.abspath(args.base_dir)

    # 2. 从 .env 文件读取 BASE_DIR
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        if key.strip() == "BASE_DIR":
                            return os.path.abspath(value.strip().strip('"\''))

    # 3. 从 config.txt 读取第一行
    config_path = Path("config.txt")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                return os.path.abspath(first_line)

    # 4. 默认当前工作目录
    return os.getcwd()

def read_local_file(file_path):
    """
    尝试读取本地文件内容，返回字符串。
    """
    print(f"正在读取: {file_path} ...", end="", flush=True)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(" [成功]")
        return content
    except Exception as e:
        print(f" [失败: {e}]")
        return f"\n> [错误：无法读取文件 - {e}]\n"

def process_file(input_path, output_path, base_dir):
    """
    读取输入文件，处理所有 [相对路径]，并写入输出文件。
    """
    if not os.path.exists(input_path):
        print(f"错误：找不到输入文件 '{input_path}'")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 匹配方括号内的任意内容（相对路径）
    pattern = r"\[([^\]]+)\]"

    def replace_callback(match):
        rel_path = match.group(1).strip()
        original_tag = match.group(0)

        # 如果路径看起来像 URL（以 http:// 或 https:// 开头），可选择保留或报错
        # 这里我们按本地文件处理，但给出提示
        if rel_path.startswith(("http://", "https://")):
            print(f"警告：发现疑似 URL 的路径 '{rel_path}'，将按本地文件尝试读取，若不存在会报错。")

        # 拼接绝对路径
        full_path = os.path.join(base_dir, rel_path)
        # 规范化路径（处理 .. 和 .）
        full_path = os.path.abspath(full_path)

        # 可选的安全检查：确保文件在 base_dir 内（防止路径遍历）
        if not full_path.startswith(os.path.abspath(base_dir)):
            error_msg = f"安全警告：路径 '{rel_path}' 尝试访问基础目录之外的区域，已阻止。"
            print(error_msg)
            return f"{original_tag}\n\n> [错误：{error_msg}]\n"

        fetched_text = read_local_file(full_path)

        # 组装新内容：保留原标记 + 换行 + 文件内容（代码块包裹） + 换行
        return f"{original_tag}\n\n```\n{fetched_text}\n```\n"

    new_content = re.sub(pattern, replace_callback, content)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"\n处理完成！结果已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="将 Markdown 中的 [相对路径] 替换为对应本地文件的内容")
    parser.add_argument("input_file", help="输入文件路径")
    parser.add_argument("output_file", help="输出文件路径")
    parser.add_argument("--base-dir", "-b", help="基础目录（绝对路径），优先级最高")
    args = parser.parse_args()

    base_dir = get_base_dir(args)
    print(f"使用基础目录: {base_dir}")

    process_file(args.input_file, args.output_file, base_dir)

if __name__ == "__main__":
    main()
