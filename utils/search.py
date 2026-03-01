#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
search - 返回搜索请求字符串
用法: py utils.py search <查询词>
"""

def run(ctx, args):
    """
    执行 search 工具。
    ctx: 上下文对象（包含根路径等，此处未使用）
    args: 命令行参数列表，如 ['hello', 'world']
    返回: 拼接好的字符串，或错误提示
    """
    if len(args) == 0:
        return "错误：请提供搜索词"
    query = ' '.join(args)
    return f"请帮我搜索 {query}"
