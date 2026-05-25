import subprocess
from loop666 import is_process_running


def test_is_process_running():
    # 当前进程自身应被检测到
    assert is_process_running("loop.py") is True

    # 不存在的进程应返回 False
    assert is_process_running("this_process_does_not_exist_xyz") is False


if __name__ == "__main__":
    test_is_process_running()
    print("OK")
