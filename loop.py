import json
import subprocess
import time
from pathlib import Path
from board_api import request_board
from opencode import Opencode

BASE_DIR = Path(__file__).parent
oc = Opencode()

REPLIED_LOG = BASE_DIR / ".replied_threads"


def load_replied() -> set:
    if REPLIED_LOG.exists():
        return set(REPLIED_LOG.read_text().strip().splitlines())
    return set()


def save_replied(no: int):
    replied = load_replied()
    replied.add(str(no))
    REPLIED_LOG.write_text("\n".join(sorted(replied)))


def is_already_replied(no: int) -> bool:
    return str(no) in load_replied()


def reply_to_topic(bid, tid, name, content):
    script = Path("/home/lumin/.claude/skills/moonchan-forum/scripts/moonchan.py")
    cmd = ["python3", str(script), "reply", str(bid), str(tid), name, content]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def process_thread(thread):
    no = thread.get("no", 0)
    tid = no
    txt = thread.get("txt", "")
    num = thread.get("num", 0)

    if is_already_replied(no):
        print(f"  [loop] no.{no} 已处理过，跳过")
        return

    if num > 0:
        for reply in thread.get("list", []):
            if "Loop666" in reply.get("n", ""):
                print(f"  [loop] no.{no} 已有 Loop666 回复，跳过")
                save_replied(no)
                return

    print(f"  [loop] 处理新需求 no.{no}: {txt[:60]}...")

    prompt = (
        f"你看到以下来自 Board 666 的需求，请分析并执行。\n\n"
        f"需求内容:\n{txt}\n\n"
        f"请完成任务后回复结果。"
    )
    try:
        result = oc.run_prompt(prompt, agent="Auto666")
        output = result.strip() if result else "任务执行完毕。"
    except Exception as e:
        output = f"执行出错: {e}"

    reply_content = (
        f"## Loop666 执行报告\n\n"
        f"检测到需求（no.{no}），执行结果如下：\n\n"
        f"{output}\n\n"
        f"#loop666 #自动执行"
    )
    reply_to_topic(666, tid, "Loop666", reply_content)
    save_replied(no)
    print(f"  [loop] no.{no} 已回复")


def main():
    print("[loop] 启动 Board 666 需求监听...")
    while True:
        print("[loop] 检查 Board 666...")
        (BASE_DIR / ".last_update").touch()

        try:
            raw = request_board(bid=666)
            data = json.loads(raw)
        except Exception as e:
            print(f"[loop] 获取失败: {e}")
            time.sleep(30)
            continue

        found = False
        for thread in data:
            if not thread.get("txt", "").strip():
                continue
            process_thread(thread)
            found = True

        if not found:
            print("[loop] Board 666 当前无帖子")

        time.sleep(60)


if __name__ == "__main__":
    main()
