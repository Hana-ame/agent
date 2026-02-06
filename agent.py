import subprocess
import time
import sys
import argparse


def run_command(cmd):
    """执行系统命令并等待结束"""
    print(f"\n[Running Command]: {' '.join(cmd)}")
    try:
        # 使用 sys.executable 确保使用当前的 python 解释器
        # 如果你的系统 'py' 命令映射到了 python，也可以直接写 "py"
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
    except KeyboardInterrupt:
        print("\n用户停止运行。")
        sys.exit(0)


def main():
    # 1. Setup the Argument Parser
    parser = argparse.ArgumentParser(
        description="Run LLM inference with configurable parameters."
    )

    # 2. Define the flags (-c, -p, -l)
    parser.add_argument(
        "-c",
        "--config",
        default="glm-9b.json",
        help="Path to the config JSON file (default: glm-9b.json)",
    )

    parser.add_argument(
        "--profiles",
        default="default",
        help="Path to the prompt file (default: default)",
    )

    parser.add_argument(
        "-p",
        "--prompt",
        default="prompt1.txt",
        help="Path to the prompt text file (default: prompt1.txt)",
    )

    parser.add_argument(
        "-l", "--loop", type=int, default=8, help="Number of loops to run (default: 8)"
    )

    # 3. Parse the CLI arguments
    args = parser.parse_args()

    # 4. Use the values in your script
    CONFIG_FILE = args.config
    PROMPT_FILE = args.prompt
    LOOP_COUNT = args.loop
    PROFILES = args.profiles

    # --- Your existing logic starts here ---
    print(f"Starting execution:")
    print(f"  Config: {CONFIG_FILE}")
    print(f"  Prompt: {PROMPT_FILE}")
    print(f"  Loops:  {LOOP_COUNT}")
    print("-" * 30)

    for i in range(LOOP_COUNT):
        iteration = 1
        while True:
            print(f"\n{'='*20} 开始第 {iteration} 轮大循环 {'='*20}")

            # 1. 第一次运行：初始化任务，读取 prompt.txt 并开启新 session (--new)
            # 注意：@prompt.txt 会将文件内容作为输入传给 kimi.py
            init_cmd = [
                "py",
                "kimi.py",
                "-p",
                f"{PROFILES}",
                "-c",
                CONFIG_FILE,
                f"@{PROMPT_FILE}",
                f"@prompt.txt",
                "--new",
            ]
            run_command(init_cmd)

            # 2. 连续运行 20 次：不带 --new，Agent 会根据 session 历史和工具修改 prompt.txt
            for i in range(1, LOOP_COUNT + 1):
                print(f"\n[子轮次 {i}/{LOOP_COUNT}]")
                loop_cmd = [
                    "py",
                    "kimi.py",
                    "-p",
                    f"{PROFILES}",
                    "-c",
                    CONFIG_FILE,
                    f"\n[子轮次 {i}/{LOOP_COUNT}]",
                ]
                run_command(loop_cmd)

                # 适当留一点间隔，防止 API 频率限制
                time.sleep(1)

            iteration += 1
            print(
                f"\n第 {iteration-1} 轮大循环结束，准备根据修改后的 {PROMPT_FILE} 开启新一轮..."
            )
            time.sleep(2)


if __name__ == "__main__":
    main()
