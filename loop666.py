import json
import subprocess
import time
from datetime import datetime, timezone
from board_api import request_board
from pathlib import Path

BASE_DIR = Path(__file__).parent


def is_process_running(name):
    result = subprocess.run(
        ["pgrep", "-f", name],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def start_loop_py():
    log_path = BASE_DIR / "loop.log"
    cmd = f"nohup python3 loop.py > {log_path} 2>&1 &"
    subprocess.Popen(cmd, shell=True)
    print("[Loop666] loop.py 已启动")


def run_restart_script():
    script = BASE_DIR / "restart_loop666.sh"
    subprocess.run(["bash", str(script)])
    print("[Loop666] restart_loop666.sh 已执行")


def reply_to_topic(bid, tid, name, content):
    script = Path("/home/lumin/.claude/skills/moonchan-forum/scripts/moonchan.py")
    cmd = [
        "python3", str(script), "reply",
        str(bid), str(tid), name, content,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def find_restart_command(data, after_ts):
    """遍历板块内容，查找包含 [restart 666] 且时间戳晚于 after_ts 的帖子。
    
    返回: (no, id, tid) 或 None
    """
    for thread in data:
        ts = thread.get("ts", "")
        no = thread.get("no", 0)
        tid = no  # thread no 即 topic ID
        txt = thread.get("txt") or ""

        if ts > after_ts and "[restart 666]" in txt:
            return (no, thread.get("id"), tid)

        # 检查回复列表
        for reply in thread.get("list", []):
            rts = reply.get("ts", "")
            rtxt = reply.get("txt") or ""
            if rts > after_ts and "[restart 666]" in rtxt:
                return (reply.get("no"), reply.get("id"), tid)

    return None


# ── 启动检查 ────────────────────────────────────────────────────────


def main():
    print("[Loop666] 检查 loop.py 进程状态...")
    if not is_process_running("loop.py"):
        print("[Loop666] loop.py 未运行，尝试启动...")
        start_loop_py()
    else:
        print("[Loop666] loop.py 已在运行")

    # 记录初始时间戳（UTC ISO 8601）
    stored_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[Loop666] 初始时间戳: {stored_ts}")

    # ── 主循环 ──────────────────────────────────────────────────────

    while True:
        print("[Loop666] 检查 Board 666...")

        try:
            raw = request_board(bid=666)
        except RuntimeError as e:
            print(f"[Loop666] 获取失败: {e}")
            time.sleep(30)
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[Loop666] JSON 解析失败: {e}")
            time.sleep(30)
            continue

        # 更新存活标记
        (BASE_DIR / ".last_update").touch()

        found = find_restart_command(data, stored_ts)

        if found:
            no, author_id, tid = found
            print(f"[Loop666] 发现 [restart 666] 指令: no={no}, tid={tid}")

            # 回复确认
            reply_text = (
                f"## Loop666 重启报告\n\n"
                f"检测到 [restart 666] 指令（no.{no}），正在执行重启...\n\n"
                f"#loop666 #重启"
            )
            reply_to_topic(666, tid, "Loop666", reply_text)
            print(f"[Loop666] 已回复 no.{tid} 确认重启")

            # 执行重启脚本（会杀掉当前 loop666 进程并启动新实例）
            run_restart_script()

            # 更新时间戳
            stored_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"[Loop666] 时间戳已更新: {stored_ts}")

            # 注意：restart_loop666.sh 会杀掉当前进程，因此不会继续执行
        else:
            print("[Loop666] 未发现 [restart 666] 指令")

        time.sleep(60)


if __name__ == "__main__":
    main()
