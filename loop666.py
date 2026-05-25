import time
import requests
from opencode import Opencode

oc = Opencode()
previous_content = None

while True:
    print("[Loop666] 检查 API...")
    try:
        response = requests.get("https://vps.moonchan.xyz/api/v2/?bid=666&tid=0&pn=0")
        current_content = response.text

        if current_content == previous_content:
            print("[Loop666] 内容无变化，跳过本次运行。")
            time.sleep(60)
            continue

        previous_content = current_content
        print("[Loop666] 内容已更新，执行 Auto666 任务...")

        result = oc.run_prompt(
            "检查 Board 666 的最新帖子，获取其中的指令并执行。然后向 Board 666 回复执行结果。",
            agent="Auto666",
        )
        print(f"[Loop666] 结果:\n{result}")

        # 在 result reporting 后再次 fetch 保存最新状态，避免 Auto666 的回复触发自身
        response = requests.get("https://vps.moonchan.xyz/api/v2/?bid=666&tid=0&pn=0")
        previous_content = response.text
        print("[Loop666] 已刷新最新状态，避免自触发。")
    except Exception as e:
        print(f"[Loop666] 错误: {e}")
    print("[Loop666] 等待 60 秒后重新检查...")
    time.sleep(60)

