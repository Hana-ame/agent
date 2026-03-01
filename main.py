
import sys
import asyncio
import os
import shlex
import subprocess
from llm_client import LLMClient

# 配置文件路径
MESSAGE_FILE = "MESSAGE.txt"
LAST_RESPONSE_FILE = "LAST_RESPONSE.txt"
SYSTEM_PROMPT_FILE = "SYSTEM_PROMPT.txt"
DEFAULT_MESSAGE_FILE = "MESSAGE_DEFAULT.txt"
PAUSE_FLAG_FILE = ".pause"
LOG_FILE = "agent.log"
FINISH_MARKER = "=== FINISH ==="  # 任务完成标记，必须在单独一行（允许前后空白）

def save_response(text: str, file=LAST_RESPONSE_FILE) -> None:
    """保存文本到指定文件，去除首尾的代码块标记（```）"""
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines.pop(0)
    if lines and lines[-1].startswith("```"):
        lines.pop()
    with open(file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def read_and_clear_message(file=MESSAGE_FILE) -> str:
    """读取文件内容并清空，若文件不存在返回空字符串"""
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
    """读取文件内容，若不存在返回空字符串"""
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
        log.write(f"\n{'='*25}\n发送的消息\n{'='*25}\n")
        log.write(sent + "\n")
        log.write(f"\n{'='*25}\n推理过程\n{'='*25}\n")
        log.write(reasoning + "\n")
        log.write(f"\n{'='*25}\n返回内容\n{'='*25}\n")
        log.write(content + "\n")
        if tool_output is not None:
            log.write(f"\n{'='*25}\n工具执行输出\n{'='*25}\n")
            log.write(tool_output + "\n")

def find_command_line(text: str) -> str | None:
    """
    在文本中查找第一个以 'py utils.py' 开头的行（忽略前后空白）。
    返回该行内容（去除两端空白），若找不到返回 None。
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("py utils.py"):
            return stripped
    return None

def is_command_present(text: str) -> bool:
    """判断文本中是否包含命令"""
    return find_command_line(text) is not None

def extract_command(text: str) -> str:
    """提取第一个命令行的内容（已 strip）"""
    return find_command_line(text) or ""

async def main():
    # 1. 初始化客户端并连接
    client = LLMClient("wss://d.810114.xyz/ws/client")
    success = await client.connect_and_pair("deepseek")
    if not success:
        print("❌ 配对失败！")
        return
    print("✅ 成功独占配对浏览器！")

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
        print("❌ 没有可发送的消息。请通过命令行参数、MESSAGE.txt 或 MESSAGE_DEFAULT.txt 提供。")
        return

    # 3. Agent 主循环
    current_msg = initial_msg
    round_num = 0

    while True:
        round_num += 1

        # 3.1 检查暂停文件 .pause
        while os.path.exists(PAUSE_FLAG_FILE):
            print("⏸️ 检测到 .pause 文件，暂停中... (等待文件被删除)")
            await asyncio.sleep(1)

        # 3.2 发送消息给 LLM
        print(f"\n✍️ 第 {round_num} 轮发送信息...")
        await client.send_prompt(current_msg)

        # 3.3 等待回复
        print("⏳ 等待 LLM 回复...\n")
        reasoning, content = await client.completion()

        # 3.4 记录日志
        log_entry(round_num, current_msg, reasoning, content)

        # 3.5 显示推理和内容
        print(f"🧠 REASONING:\n{reasoning}\n")
        print(f"📝 CONTENT:\n{content}\n")

        # 3.6 检查是否包含单独成行的结束标记
        lines = content.splitlines()
        # 找出所有单独成行的标记（允许前后空白）
        finish_lines = [i for i, line in enumerate(lines) if line.strip() == FINISH_MARKER]
        if finish_lines:
            print("✅ 检测到单独成行的任务完成标记，结束循环。")
            # 去除所有标记行（可能多个，但通常只有一个）
            filtered_lines = [line for i, line in enumerate(lines) if i not in finish_lines]
            final_content = "\n".join(filtered_lines).strip()
            save_response(final_content, LAST_RESPONSE_FILE)
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write("\n【最终答案】已保存到 LAST_RESPONSE.txt\n")
            break

        # 3.7 检查是否有命令
        if is_command_present(content):
            # 提取第一个命令
            command = extract_command(content)
            print(f"⚙️ 执行命令: {command}")

            # 准备执行命令
            try:
                # 分割命令，并将 'py' 替换为当前 Python 解释器路径
                parts = shlex.split(command)
                if parts and parts[0] in ('py', 'python', 'python3'):
                    parts[0] = sys.executable
                result = subprocess.run(parts, capture_output=True, text=True, cwd=os.getcwd())
                output = result.stdout + result.stderr
                if result.returncode != 0:
                    output = f"命令执行失败 (返回码 {result.returncode}):\n{output}"
                else:
                    output = output.strip()
            except Exception as e:
                output = f"执行命令时发生异常: {str(e)}"

            print(f"📤 命令输出:\n{output}")

            # 记录工具输出到日志
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write("\n【工具执行输出】\n")
                log.write(output + "\n")

            # 处理暂停情况：如果输出以 PAUSED: 开头，进入等待模式
            if output.startswith("PAUSED:"):
                print("⏸️ 检测到暂停请求，进入人工干预等待...")
                # 清空 MESSAGE.txt，避免残留
                with open(MESSAGE_FILE, "w", encoding="utf-8") as f:
                    f.write("")
                # 等待 .pause 文件被删除（用户执行 resume 或手动删除）
                while os.path.exists(PAUSE_FLAG_FILE):
                    await asyncio.sleep(1)
                print("▶️ 恢复运行")
                # 恢复后，检查用户是否在 MESSAGE.txt 中写入了新指令
                current_msg = read_and_clear_message(MESSAGE_FILE)
                if not current_msg:
                    # 提供默认恢复提示
                    current_msg = "用户已执行恢复，请继续任务。"
                continue  # 直接进入下一轮，不再将 output 作为下一轮消息
            else:
                # 正常命令输出，写入 MESSAGE.txt 作为下一轮输入
                with open(MESSAGE_FILE, "w", encoding="utf-8") as f:
                    f.write(output)
                current_msg = output
        else:
            # 没有命令，也没有结束标记，视为中间内容（如需要写入文件的内容）
            print("📝 收到非命令内容，保存到 LAST_RESPONSE.txt 并继续...")
            save_response(content, LAST_RESPONSE_FILE)
            # 提供默认提示，让 LLM 知道内容已保存
            default_next = "内容已保存到 LAST_RESPONSE.txt，请根据情况继续。"
            with open(MESSAGE_FILE, "w", encoding="utf-8") as f:
                f.write(default_next)
            current_msg = default_next

    print("👋 Agent 结束。")
    # 可选关闭连接
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())