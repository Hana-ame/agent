import trace
import sys

def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

# 创建追踪器，统计行执行次数
tracer = trace.Trace(count=1, trace=0)
tracer.run('fib(5)')

# 报告统计
results = tracer.results()
results.write_results(show_missing=True, coverdir='.')
print("Trace results written to .cover files")

import os
os.remove('trace_advanced.cover')  # 清理临时文件
