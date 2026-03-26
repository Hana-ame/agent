import resource
import sys

if sys.platform.startswith('linux') or sys.platform == 'darwin':
    # 获取软限制
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"File descriptor limit: soft={soft}, hard={hard}")
    # 设置新的软限制（仅演示）
    # resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    print("Resource module works on this system.")
else:
    print("resource module not fully supported on this platform")
