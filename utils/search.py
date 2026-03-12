#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
search - 返回搜索请求字符串

用法：py utils.py search <查询词>

参数：
  <查询词>  一个或多个单词，自动拼接为搜索请求。

成功输出：直接返回拼接后的请求字符串。
失败输出：
  === search ===
  错误：具体错误信息
  === end of search ===
"""

def run(ctx, args):
    if len(args) == 0:
        return "=== search ===\n错误：请提供搜索词\n=== end of search ==="
    query = ' '.join(args)
    return f"请帮我搜索 {query}"