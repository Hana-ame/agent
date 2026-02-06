import subprocess
import time
import sys

# 配置参数
CONFIG_FILE = "qwen3-8b.json"
CONFIG_FILE = "glm-9b.json"
PROMPT_FILE = "prompt1.txt"
LOOP_COUNT = 8  # 每一轮次运行的次数

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
    iteration = 1
    while True:
        print(f"\n{'='*20} 开始第 {iteration} 轮大循环 {'='*20}")

        # 1. 第一次运行：初始化任务，读取 prompt.txt 并开启新 session (--new)
        # 注意：@prompt.txt 会将文件内容作为输入传给 kimi.py
        init_cmd = ["py", "kimi.py", "-c", CONFIG_FILE, f"@{PROMPT_FILE}", f"@prompt.txt", "--new"]
        run_command(init_cmd)

        # 2. 连续运行 20 次：不带 --new，Agent 会根据 session 历史和工具修改 prompt.txt
        for i in range(1, LOOP_COUNT + 1):
            print(f"\n[子轮次 {i}/{LOOP_COUNT}]")
            loop_cmd = ["py", "kimi.py", "-c", CONFIG_FILE, f"\n[子轮次 {i}/{LOOP_COUNT}]"]
            run_command(loop_cmd)
            
            # 适当留一点间隔，防止 API 频率限制
            time.sleep(1)

        iteration += 1
        print(f"\n第 {iteration-1} 轮大循环结束，准备根据修改后的 {PROMPT_FILE} 开启新一轮...")
        time.sleep(2)

if __name__ == "__main__":
    main()