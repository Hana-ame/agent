import time
from pathlib import Path
from opencode import Opencode

BASE_DIR = Path(__file__).parent
oc = Opencode()


def main():
    print("[loop] 启动 Board 666 监听...")
    while True:
        print("[loop] 触发 Auto666 检查 Board 666...")
        (BASE_DIR / ".last_update").touch()

        try:
            result = oc.run_prompt(
                "检查 Board 666 的最新帖子，找出未处理的需求，将代码需求写成 Python 脚本并执行，回复结果。",
                agent="Auto666",
            )
            print(f"[loop] Auto666 返回:\n{result[:500]}")
        except Exception as e:
            print(f"[loop] 执行出错: {e}")

        time.sleep(60)


if __name__ == "__main__":
    main()
