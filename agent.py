import sys
import asyncio
import os
import shlex
import subprocess
import argparse
from pathlib import Path
from llm_client import LLMClient

def get_root_path():
    """从.env文件读取ROOT_PATH，若不存在则返回当前工作目录"""
    root = os.getcwd()
    env_path = os.path.join(os.getcwd(), '.env')
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('ROOT_PATH'):
                    root = line.strip().split('=', 1)[1].strip()
                    # 去除可能的引号
                    if root.startswith('"') and root.endswith('"'):
                        root = root[1:-1]
                    elif root.startswith("'") and root.endswith("'"):
                        root = root[1:-1]
                    break
    except FileNotFoundError:
        pass
    return os.path.abspath(root)

def get_utils_path():
    """从.env文件读取ROOT_PATH，若不存在则返回当前工作目录"""
    root = os.getcwd()
    env_path = os.path.join(os.getcwd(), '.env')
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('UTILS_PATH'):
                    root = line.strip().split('=', 1)[1].strip()
                    # 去除可能的引号
                    if root.startswith('"') and root.endswith('"'):
                        root = root[1:-1]
                    elif root.startswith("'") and root.endswith("'"):
                        root = root[1:-1]
                    break
    except FileNotFoundError:
        pass
    return os.path.abspath(root)

# 根目录
ROOT_PATH = get_root_path()
UTILS_PATH = get_utils_path()

# 所有文件路径均基于 ROOT_PATH
MESSAGE_FILE = os.path.join(ROOT_PATH, "MESSAGE.txt")
LAST_RESPONSE_FILE = os.path.join(ROOT_PATH, "LAST_RESPONSE.txt")
# 系统提示文件现在放在 agent 子目录下，与 profiles.json 同位置
SYSTEM_PROMPT_FILE = os.path.join(ROOT_PATH, "SYSTEM_PROMPT.txt")
DEFAULT_MESSAGE_FILE = os.path.join(ROOT_PATH, "MESSAGE_DEFAULT.txt")
PAUSE_FLAG_FILE = os.path.join(ROOT_PATH, ".pause")
LOG_FILE = os.path.join(ROOT_PATH, "agent.log")
FINISH_MARKER = "=== FINISH ==="
PROFILES_PATH = os.path.join(ROOT_PATH, "agent", "profiles.json")
PAYLOADS_DIR = os.path.join(ROOT_PATH, "payloads")

def save_response(text: str, file=LAST_RESPONSE_FILE) -> None:
    """保存响应内容到文件，移除首尾的```标记"""
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines.pop(0)
    if lines and lines[-1].startswith("```"):
        lines.pop()
    with open(file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def read_and_clear_message(file=MESSAGE_FILE) -> str:
    """读取并清空消息文件"""
    content = ""
    try:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        with open(file, "w", encoding="utf-8") as f:
            f.write("")
    except FileNotFoundError:
        pass
    return content

def read_file_content(file: str) -> str:
    """读取文件内容并去除首尾空白"""
    try:
        with open(file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""

def log_entry(round_num: int, sent: str, reasoning: str, content: str, tool_output: str = None):
    """记录一轮交互到日志文件"""
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n{'='*50}\n")
        log.write(f"第 {round_num} 轮\n")
        log.write(f"{'='*50}\n")
        log.write(f"\n{'='*25}\n发送内容\n{'='*25}\n")
        log.write(sent + "\n")
        log.write(f"\n{'='*25}\n推理过程\n{'='*25}\n")
        log.write(reasoning + "\n")
        log.write(f"\n{'='*25}\n回复内容\n{'='*25}\n")
        log.write(content + "\n")
        if tool_output is not None:
            log.write(f"\n{'='*25}\n工具输出\n{'='*25}\n")
            log.write(tool_output + "\n")

def find_command_line(text: str) -> str | None:
    """
    查找以 'py utils.py' 开头的行，返回第一个匹配行（已strip）
    如果没有找到返回 None
    """
    for line in text.splitlines():
        if line.startswith("py utils.py"):
            return line
        # 误判率过高。
        # stripped = line.strip()
        # if stripped.startswith("py utils.py"):
        #     return stripped
    return None

def is_command_present(text: str) -> bool:
    """判断回复中是否包含命令"""
    return find_command_line(text) is not None

def extract_command(text: str) -> str:
    """提取第一条命令（已strip）"""
    return find_command_line(text) or ""

async def main_async(args):
    """
    异步主函数
    args: 解析后的参数对象
    """
    print(f"当前工作目录: {ROOT_PATH}")
    print(f"当前工具目录: {UTILS_PATH}")
    print(f"连接参数: {args.connection}")
    if not args.connection.startswith(("ws://", "wss://")):
        print(f"使用 payload: {args.payload}")

    # 1. 初始化客户端（自动根据参数选择后端）
    client = LLMClient(
        connection_param=args.connection,
        payload_name=args.payload,
        profiles_path=Path(PROFILES_PATH),
        root_path=Path(ROOT_PATH),
    )
    success = await client.connect()
    if not success:
        print("连接失败")
        return
    print("连接成功，开始任务")

    # 2. 构建初始消息
    system_prompt = read_file_content(SYSTEM_PROMPT_FILE)
    user_msg = args.message if args.message else read_and_clear_message(MESSAGE_FILE)
    if not user_msg:
        user_msg = read_file_content(DEFAULT_MESSAGE_FILE)

    if system_prompt and user_msg:
        initial_msg = f"{system_prompt}\n\n{user_msg}"
    elif system_prompt:
        initial_msg = system_prompt
    else:
        initial_msg = user_msg

    if not initial_msg:
        print("没有初始消息，请提供命令行参数或在MESSAGE.txt/MESSAGE_DEFAULT.txt中写入内容")
        return

    # 3. Agent 主循环
    current_msg = initial_msg
    round_num = 0

    while True:
        # defeault output
        output = "上一轮对话中的回复内容已保存到 LAST_RESPONSE.txt，如果需要保存，请请根据情况使用py utils.py write或者py utils.py write_multiple。如果输出的不是完整代码或内容中包含代码以外的说明，请先输出完整的，不带说明的代码（注释是被允许的）。"

        round_num += 1

        # 3.1 检查暂停文件
        while os.path.exists(PAUSE_FLAG_FILE):
            print("检测到 .pause 文件，暂停中... (等待删除)")
            await asyncio.sleep(1)

        # 3.2 发送消息给 LLM
        print(f"\n{'='*50}\n第 {round_num} 轮：发送消息...")
        await client.send_prompt(current_msg)

        # 3.3 接收回复
        print("等待 LLM 回复...\n")
        reasoning, content = await client.completion()

        # 3.4 记录日志
        log_entry(round_num, current_msg, reasoning, content)

        # 3.5 显示回复
        print("="*30)
        print(f"推理过程:\n{reasoning}\n")
        print("="*30)
        print(f"回复内容:\n{content}\n")

        # 3.6 检查结束标记
        lines = content.splitlines()
        finish_lines = [i for i, line in enumerate(lines) if line.strip() == FINISH_MARKER]
        if finish_lines:
            print("检测到结束标记，任务完成")
            # 移除结束标记行后保存最终答案
            filtered_lines = [line for i, line in enumerate(lines) if i not in finish_lines]
            final_content = "\n".join(filtered_lines).strip()
            save_response(final_content, LAST_RESPONSE_FILE)
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write("\n最终答案已保存到 LAST_RESPONSE.txt\n")
            break

        # 3.7 处理命令
        if is_command_present(content):
            command = extract_command(content)
            print("="*30)
            print(f"执行命令: {command}")

            try:
                parts = shlex.split(command)
                if parts and parts[0] in ('py', 'python', 'python3'):
                    parts[0] = sys.executable
                    # 如果第二个参数是 utils.py 或 ./utils.py，则替换为绝对路径
                    if len(parts) > 1 and os.path.basename(parts[1]) == 'utils.py':
                        parts[1] = os.path.join(UTILS_PATH, 'utils.py')
                # 然后设置 cwd=ROOT_PATH
                result = subprocess.run(parts, capture_output=True, text=True, cwd=ROOT_PATH)
                output = result.stdout + result.stderr
                if result.returncode != 0:
                    output = f"命令执行失败 (返回码 {result.returncode}):\n{output}"
                else:
                    output = output.strip()
            except Exception as e:
                output = f"执行命令时出错: {str(e)}"

            print("="*30)
            print(f"工具输出:\n{output}")

            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write("\n工具输出\n")
                log.write(output + "\n")

            if output.startswith("PAUSED:"):
                print("工具请求暂停，进入人工干预...")
                # 清空 MESSAGE.txt，等待用户恢复
                with open(MESSAGE_FILE, "w", encoding="utf-8") as f:
                    f.write("")
                while os.path.exists(PAUSE_FLAG_FILE):
                    await asyncio.sleep(1)
                print("暂停解除")
                current_msg = read_and_clear_message(MESSAGE_FILE)
                if not current_msg:
                    current_msg = "人工干预结束，请继续任务"
                continue
            else:
                with open(MESSAGE_FILE, "w", encoding="utf-8") as f:
                    f.write(output)
                current_msg = output
        # 你这样子做了会出问题的。
        # else:
        #     # 无命令，保存回复内容到 LAST_RESPONSE.txt，并提示继续
        #     print("回复中无命令，已保存到 LAST_RESPONSE.txt")
        
        save_response(content, LAST_RESPONSE_FILE)
        current_msg = output

    print("="*30)
    print("Agent 结束")
    await client.close()

def main():
    """入口函数，解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Agent 客户端，支持 WebSocket 和 HTTP 后端"
    )
    parser.add_argument(
        "connection",
        help="连接参数：WebSocket URL (以 ws:// 或 wss:// 开头) 或 profiles.json 中的键名 (如 silicon)"
    )
    parser.add_argument(
        "-m", "--message",
        help="直接提供消息内容，否则从 MESSAGE.txt 或 MESSAGE_DEFAULT.txt 读取"
    )
    parser.add_argument(
        "-p", "--payload",
        default="default.json",
        help="payload 文件名（位于 payloads/ 目录下），仅用于 HTTP 模式，默认 default.json"
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()