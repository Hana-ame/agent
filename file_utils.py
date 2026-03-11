import os
import shutil
import sys
import asyncio
from pathlib import Path

# ==================== 配置常量 ====================
MAX_RECONNECT_ATTEMPTS = 3          # 最大重连尝试次数
RECONNECT_DELAY = 2                  # 重连延迟（秒）
COMPLETION_TIMEOUT = 300              # 单次 completion 超时时间（秒）
COMMAND_TIMEOUT = 60                   # 执行外部命令的超时时间（秒）

# ==================== 路径工具 ====================
def get_root_path():
    """从.env文件读取ROOT_PATH，若不存在则返回当前工作目录"""
    root = os.getcwd()
    env_path = os.path.join(os.getcwd(), ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("ROOT_PATH"):
                    root = line.strip().split("=", 1)[1].strip()
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

# 文件路径常量
MESSAGE_FILE = os.path.join(ROOT_PATH, "MESSAGE.txt")
LAST_RESPONSE_FILE = os.path.join(ROOT_PATH, "LAST_RESPONSE.txt")
THIS_RESPONSE_FILE = os.path.join(ROOT_PATH, "THIS_RESPONSE.txt")
SYSTEM_PROMPT_FILE = os.path.join(ROOT_PATH, "SYSTEM_PROMPT.txt")
DEFAULT_MESSAGE_FILE = os.path.join(ROOT_PATH, "DEFAULT_MESSAGE.txt")
PAUSE_FLAG_FILE = os.path.join(ROOT_PATH, ".pause")
LOG_FILE = os.path.join(ROOT_PATH, "agent.log")
FINISH_MARKER = "=== FINISH ==="
# 修改：将 profiles.json 放在根目录，避免创建 agent 子目录
PROFILES_PATH = os.path.join(ROOT_PATH, "profiles.json")
PAYLOADS_DIR = os.path.join(ROOT_PATH, "payloads")

# ==================== 文件操作工具 ====================
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
    """构造初始消息（基于 --new-chat 标志）"""
    system_prompt = read_file_content(SYSTEM_PROMPT_FILE)
    user_msg = args.message if args.message else read_file_content(DEFAULT_MESSAGE_FILE)

    if args.new_chat:
        if system_prompt and user_msg:
            initial_msg = f"{system_prompt}\n\n{user_msg}"
        elif system_prompt:
            initial_msg = system_prompt
        else:
            initial_msg = user_msg
    else:
        initial_msg = user_msg

    return initial_msg


def read_system_prompt():
    """
    读取 .agent/system_prompt.txt 文件的内容并返回。
    假设文件采用 UTF-8 编码。
    """
    with open('.agent/system_prompt.txt', 'r', encoding='utf-8') as file:
        return file.read()


def save_agent_content(self, content):
    """
    保存新内容到 THIS_CONTENT.txt，并将旧内容滚动到 LAST_CONTENT.txt
    """
    # 1. 如果当前内容文件已存在，将其备份到 LAST_CONTENT.txt
    if os.path.exists(self.this_response_file):
        # 使用 copyfile 或 move，取决于你是否想保留 THIS 文件
        shutil.copyfile(self.this_response_file, self.last_response_file)
    
    # 2. 将新内容写入 THIS_CONTENT.txt
    try:
        with open(self.this_response_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"内容已成功保存至 {self.this_response_file}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

# 调用示例
# content = "这是本次对话的内容..."
# self.save_agent_content(content)