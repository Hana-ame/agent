import sys
import asyncio
import os
import shlex
import subprocess
from llm_client import LLMClient

def get_root_path():
    """从.env文件读取ROOT_PATH，若不存在则返回当前工作目录"""
    root = os.getcwd()
    env_path = os.path.join(os.getcwd(), '.env')
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('ROOT_PATH='):
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

# 所有文件路径均基于 ROOT_PATH
MESSAGE_FILE = os.path.join(ROOT_PATH, "MESSAGE.txt")
LAST_RESPONSE_FILE = os.path.join(ROOT_PATH, "LAST_RESPONSE.txt")
SYSTEM_PROMPT_FILE = os.path.join(ROOT_PATH, "SYSTEM_PROMPT.txt")
DEFAULT_MESSAGE_FILE = os.path.join(ROOT_PATH, "MESSAGE_DEFAULT.txt")
PAUSE_FLAG_FILE = os.path.join(ROOT_PATH, ".pause")
LOG_FILE = os.path.join(ROOT_PATH, "agent.log")
FINISH_MARKER = "=== FINISH ==="

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
        # stripped = line.strip()
        stripped = line
        if stripped.startswith("py utils.py"):
            return stripped
    return None

def is_command_present(text: str) -> bool:
    """判断回复中是否包含命令"""
    return find_command_line(text) is not None

def extract_command(text: str) -> str:
    """提取第一条命令（已strip）"""
    return find_command_line(text) or ""

async def main():
    print(ROOT_PATH) # 修改为print绝对路径
    # 1. 连接 LLM
    client = LLMClient("wss://d.810114.xyz/ws/client")
    success = await client.connect_and_pair("deepseek")
    if not success:
        print("连接失败")
        return
    print("连接成功，开始任务")

    # 2. 构建初始消息
    system_prompt = read_file_content(SYSTEM_PROMPT_FILE)
    user_msg = ""
    if len(sys.argv) > 1:
        user_msg = sys.argv[1]
    if not user_msg:
        user_msg = read_and_clear_message(MESSAGE_FILE)
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
        round_num += 1

        # 3.1 检查暂停文件
        while os.path.exists(PAUSE_FLAG_FILE):
            print("检测到 .pause 文件，暂停中... (等待删除)")
            await asyncio.sleep(1)

        # 3.2 发送消息给 LLM
        print(f"\n第 {round_num} 轮：发送消息...")
        await client.send_prompt(current_msg)

        # 3.3 接收回复
        print("等待 LLM 回复...\n")
        reasoning, content = await client.completion()

        # 3.4 记录日志
        log_entry(round_num, current_msg, reasoning, content)

        # 3.5 显示回复
        print(f"推理过程:\n{reasoning}\n")
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
            print(f"执行命令: {command}")

            try:
                parts = shlex.split(command)
                if parts and parts[0] in ('py', 'python', 'python3'):
                    parts[0] = sys.executable
                # 设置工作目录为 ROOT_PATH，确保工具在正确路径下运行
                result = subprocess.run(parts, capture_output=True, text=True, cwd=ROOT_PATH)
                output = result.stdout + result.stderr
                if result.returncode != 0:
                    output = f"命令执行失败 (返回码 {result.returncode}):\n{output}"
                else:
                    output = output.strip()
            except Exception as e:
                output = f"执行命令时出错: {str(e)}"

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
        else:
            # 无命令，保存回复内容到 LAST_RESPONSE.txt，并提示继续
            print("回复中无命令，已保存到 LAST_RESPONSE.txt")
            save_response(content, LAST_RESPONSE_FILE)
            default_next = "内容已保存到 LAST_RESPONSE.txt，请根据情况继续。"
            with open(MESSAGE_FILE, "w", encoding="utf-8") as f:
                f.write(default_next)
            current_msg = default_next

    print("Agent 结束")
    # await client.close()

if __name__ == "__main__":
    asyncio.run(main())