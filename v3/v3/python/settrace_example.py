import sys

def trace_calls(frame, event, arg):
    if event == 'call':
        print(f"Calling {frame.f_code.co_name}")
    elif event == 'line':
        print(f"Line {frame.f_lineno} in {frame.f_code.co_name}")
    return trace_calls

def foo():
    x = 1
    y = 2
    return x + y

sys.settrace(trace_calls)
foo()
sys.settrace(None)
