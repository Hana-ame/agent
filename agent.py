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
    env_path = os.path.join(os.getcwd(), ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("ROOT_PATH"):
                    root = line.strip().split("=", 1)[1].strip()
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
    """从.env文件读取UTILS_PATH，若不存在则返回当前工作目录"""
    root = os.getcwd()
    env_path = os.path.join(os.getcwd(), ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("UTILS_PATH"):
                    root = line.strip().split("=", 1)[1].strip()
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
THIS_RESPONSE_FILE = os.path.join(ROOT_PATH, "THIS_RESPONSE.txt")
# 系统提示文件现在放在 agent 子目录下，与 profiles.json 同位置
SYSTEM_PROMPT_FILE = os.path.join(ROOT_PATH, "SYSTEM_PROMPT.txt")
DEFAULT_MESSAGE_FILE = os.path.join(ROOT_PATH, "DEFAULT_MESSAGE.txt")
PAUSE_FLAG_FILE = os.path.join(ROOT_PATH, ".pause")
LOG_FILE = os.path.join(ROOT_PATH, "agent.log")
FINISH_MARKER = "=== FINISH ==="
PROFILES_PATH = os.path.join(ROOT_PATH, "agent", "profiles.json")
PAYLOADS_DIR = os.path.join(ROOT_PATH, "payloads")


def save_response(text: str, file=LAST_RESPONSE_FILE) -> None:
    """保存响应内容到文件，不移除任何标记"""
    with open(file, "w", encoding="utf-8") as f:
        f.write(text)


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


def log_entry(
    round_num: int, sent: str, reasoning: str, content: str, tool_output: str = None
):
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


def find_all_commands(text: str) -> list[str]:
    """
    查找所有以 'py utils.py' 开头的行（忽略行首空白），返回列表。
    """
    commands = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("py utils.py"):
            commands.append(stripped)
    return commands


def is_command_present(text: str) -> bool:
    """判断回复中是否包含任何命令"""
    return len(find_all_commands(text)) > 0


def extract_commands(text: str) -> list[str]:
    """提取所有命令（已strip）"""
    return find_all_commands(text)

def initial_message(args) -> str:
    # 2. 读取系统提示和用户消息
    system_prompt = read_file_content(SYSTEM_PROMPT_FILE)
    user_msg = args.message if args.message else read_file_content(DEFAULT_MESSAGE_FILE)

    print(args.new_chat , system_prompt, user_msg)
    
    # 3. 根据 --new-chat 标志决定是否重置对话并发送系统提示
    if args.new_chat:
        if system_prompt and user_msg:
            initial_msg = f"{system_prompt}\n\n{user_msg}"
        elif system_prompt:
            initial_msg = system_prompt
        else:
            initial_msg = user_msg
    else:
        # 不重置对话，也不发送系统提示
        initial_msg = user_msg

    return initial_msg


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

    if args.new_chat:
        await client.new_chat()
        
    if os.path.exists(PAUSE_FLAG_FILE):
        os.remove(PAUSE_FLAG_FILE)  # 删除暂停标记

    # 4. Agent 主循环
    round_num = 0

    ongoing = True
    paused = False
    while ongoing:
        
        if paused:
            print("删除.pause以继续")
        while paused:
            await asyncio.sleep(1)
        
        # 默认输出（仅当无命令且不暂停时会被使用，但新逻辑中无命令直接暂停，所以这里可保留原值，但实际不再使用）
        # output = "上一轮对话中的回复内容已保存到 LAST_RESPONSE.txt，如果需要保存，请根据情况使用 py utils.py write（对直接给出的文件）或者 py utils.py write_multiple（通过===分割的文件）进行写入。如果输出的不是完整代码或内容中包含代码以外的说明，请先输出完整的，不带说明的代码（注释是被允许的）。\n到这条信息为止，没有任何文件被保存，如果这不符合期望，请再次检查。\n提示：\npy utils.py help pause\npy utils.py help write\npy utils.py help write_multiple"
        current_msg = read_and_clear_message(MESSAGE_FILE) or initial_message(args=args)
        round_num += 1

        # 4.1 检查暂停文件
        if os.path.exists(PAUSE_FLAG_FILE):
            paused = True

        # 4.2 发送消息给 LLM
        print(f"\n{'='*50}\n第 {round_num} 轮：发送消息...")
        await client.send_prompt(current_msg)

        # 4.3 接收回复
        print("等待 LLM 回复...\n")
        reasoning, content = await client.completion()
        save_response(content, THIS_RESPONSE_FILE)

        # 4.4 记录日志
        log_entry(round_num, current_msg, reasoning, content)

        # 4.5 显示回复
        print("=" * 30)
        print(f"推理过程:\n{reasoning}\n")
        print("=" * 30)
        print(f"回复内容:\n{content}\n")

        # 4.7 处理命令
        if is_command_present(content):
            commands = extract_commands(content)
            print("=" * 30)
            print(f"检测到 {len(commands)} 条命令，开始执行...")

            all_outputs = []
            for cmd in commands:
                print(f"执行命令: {cmd}")
                try:
                    parts = shlex.split(cmd)
                    if parts and parts[0] in ("py", "python", "python3"):
                        parts[0] = sys.executable
                        # 如果第二个参数是 utils.py 或 ./utils.py，则替换为绝对路径
                        if len(parts) > 1 and os.path.basename(parts[1]) == "utils.py":
                            parts[1] = os.path.join(UTILS_PATH, "utils.py")
                    # 设置 cwd=ROOT_PATH
                    result = subprocess.run(
                        parts, capture_output=True, text=True, cwd=ROOT_PATH
                    )
                    cmd_output = result.stdout + result.stderr
                    if result.returncode != 0:
                        cmd_output = f"命令执行失败 (返回码 {result.returncode}):\n{cmd_output}"
                    else:
                        cmd_output = cmd_output.strip()
                except Exception as e:
                    cmd_output = f"执行命令时出错: {str(e)}"

                # 添加提示行
                hint = f"command {cmd} has been run, the result returned:"
                if cmd_output:
                    block = f"{hint}\n{cmd_output}"
                else:
                    block = hint + " (no output)"
                all_outputs.append(block)

            # 合并所有命令的输出，用空行分隔
            output = "\n\n".join(all_outputs)

            print("=" * 30)
            print(f"工具输出汇总:\n{output}")

            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"\n{'='*25}\n工具输出\n{'='*25}\n")
                log.write(output + "\n")

            # 将命令输出写入 MESSAGE.txt 作为下一轮的输入
            with open(MESSAGE_FILE, "w", encoding="utf-8") as f:
                f.write(output)
            # current_msg 将在循环末尾设置
        else:
            # 无命令：返回系统提示
            print("未检测到命令，返回系统提示。")
            output = initial_message(args=args)
            with open(MESSAGE_FILE, "w", encoding="utf-8") as f:
                f.write(output)

        # 保存本轮 LLM 回复到 LAST_RESPONSE.txt
        save_response(content, LAST_RESPONSE_FILE)
        # 设置下一轮的消息
        current_msg = output

    print("=" * 30)
    print("Agent 结束")
    await client.close()


def main():
    """入口函数，解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Agent 客户端，支持 WebSocket 和 HTTP 后端"
    )
    parser.add_argument(
        "connection",
        help="连接参数：WebSocket URL (以 ws:// 或 wss:// 开头) 或 profiles.json 中的键名 (如 silicon)",
    )
    parser.add_argument(
        "-m",
        "--message",
        help="直接提供消息内容，否则从 MESSAGE.txt 或 MESSAGE_DEFAULT.txt 读取",
    )
    parser.add_argument(
        "-p",
        "--payload",
        default="default.json",
        help="payload 文件名（位于 payloads/ 目录下），仅用于 HTTP 模式，默认 default.json",
    )
    parser.add_argument(
        "--new-chat",
        action="store_true",
        help="开启新对话并发送系统提示（如果存在），否则不重置对话也不发送系统提示",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
