import subprocess
import time
import sys
import argparse
import os


def run_command(cmd):
    """执行系统命令"""
    print(f"\n[Running]: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"执行失败: {e}")
    except KeyboardInterrupt:
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="LLM Agent Orchestrator")
    parser.add_argument("-c", "--config", default="qwen2.5-7b.json")
    parser.add_argument("-p", "--profiles", default="default")
    parser.add_argument(
        "-f", "--file", default="base_goal.txt", help="初始全局目标文件"
    )
    parser.add_argument("-i", "--id", default="agent_01", help="当前 Agent 的 ID")
    parser.add_argument("-l", "--loop", type=int, default=10, help="小循环次数")
    parser.add_argument("-t", "--time", type=int, default=300, help="结束后的delay")

    args = parser.parse_args()

    # 路径配置
    BASE_GOAL_FILE = args.file  # 永远不变的核心指令 (e.g., "你是一个版面管理员")
    AGENT_ID = args.id
    STATE_FILE = f"prompt_{AGENT_ID}.txt"  # 由 Tool (update_next_prompt) 维护的动态进度

    print(f"Starting Agent: {AGENT_ID}")
    print(f"Base Goal: {BASE_GOAL_FILE}")
    print("-" * 30)

    # 如果状态文件不存在，初始化一个空的
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            f.write("任务初始化：尚未开始任何操作。")

    iteration = 1
    while True:
        print(f"\n{'#'*30}\n# 开始第 {iteration} 轮大迭代 (Session Reset)\n{'#'*30}")

        # 1. 大迭代初始化：带上 --new 开启新会话
        # 同时传入：1.全局目标(@BASE_GOAL_FILE) 2.当前Agent身份 3.Tool记录的进度(@STATE_FILE)
        init_cmd = [
            "py",
            "kimi.py",
            "-p",
            args.profiles,
            "-c",
            args.config,
            f"@{BASE_GOAL_FILE}",  # 保持不变的原始 Prompt
            f"Your Agent ID is: {AGENT_ID}",
            f"Current State:",
            "@{STATE_FILE}",  # Tool 写入的进度
            "--new",  # 开启全新 Session
        ]
        run_command(init_cmd)

        # 2. 小迭代：在同一个 Session 中持续推进
        # 在这里不带 --new，Agent 会记得上面的对话历史
        for i in range(1, args.loop + 1):
            print(f"\n[小迭代 {i}/{args.loop} - Agent: {AGENT_ID}]")

            # 每次小迭代可以注入当前的时间戳或进度提醒
            loop_cmd = [
                "py",
                "kimi.py",
                "-p",
                args.profiles,
                "-c",
                args.config,
                f"Continue. (Iteration {i}/{args.loop})",  # 简单的推进指令
            ]
            run_command(loop_cmd)

            # 留出冷却时间
            time.sleep(60)

        iteration += 1
        print(f"\n本轮 Session 结束。下一轮将重新加载由 Tool 更新后的 {STATE_FILE}")
        time.sleep(args.time)
        os.system("rm chat_*")


if __name__ == "__main__":
    main()
