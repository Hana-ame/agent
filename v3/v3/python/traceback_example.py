import traceback
import sys

def cause_error():
    raise ValueError("Something went wrong")

def wrapper():
    cause_error()

try:
    wrapper()
except Exception:
    exc_type, exc_value, exc_tb = sys.exc_info()
    formatted = traceback.format_exception(exc_type, exc_value, exc_tb)
    print("Traceback:")
    print(''.join(formatted))
    
    # 打印最后一行
    print("Last line:", traceback.format_exc().splitlines()[-1])
