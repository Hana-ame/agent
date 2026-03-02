import sys
import asyncio
import os
import argparse
from pathlib import Path

from llm_client import LLMClient
import file_utils
from command_executor import CommandExecutor


class Agent:
    """Agent 主控类，管理连接、重试、主循环"""

    def __init__(self, args):
        self.args = args
        self.round_num = 0
        self.paused = False
        self.client: LLMClient = None
        self._reconnect_attempts = 0

    async def _create_client(self) -> bool:
        """创建并连接客户端，返回是否成功"""
        print(f"当前工作目录: {file_utils.ROOT_PATH}")
        print(f"当前工具目录: {file_utils.UTILS_PATH}")
        print(f"连接参数: {self.args.connection}")
        if not self.args.connection.startswith(("ws://", "wss://")):
            print(f"使用 payload: {self.args.payload}")

        client = LLMClient(
            connection_param=self.args.connection,
            payload_name=self.args.payload,
            profiles_path=Path(file_utils.PROFILES_PATH),
            root_path=Path(file_utils.ROOT_PATH),
        )
        success = await client.connect()
        if success:
            self.client = client
            self._reconnect_attempts = 0
            if self.args.new_chat:
                await client.new_chat()
            print("连接成功，开始任务")
        else:
            print("连接失败")
        return success

    async def _ensure_connected(self) -> bool:
        """确保客户端已连接且可用，若断开则尝试重连"""
        # 修复：直接访问 is_finished 属性，而不是用 getattr
        if self.client and not self.client.is_finished:
            return True
        if self.client:
            await self.client.close()
        while self._reconnect_attempts < file_utils.MAX_RECONNECT_ATTEMPTS:
            self._reconnect_attempts += 1
            print(f"尝试重连 ({self._reconnect_attempts}/{file_utils.MAX_RECONNECT_ATTEMPTS})...")
            if await self._create_client():
                return True
            await asyncio.sleep(file_utils.RECONNECT_DELAY * self._reconnect_attempts)
        print("重连失败，退出")
        return False

    async def _safe_completion(self) -> tuple[str, str]:
        """
        安全地获取 completion，包含超时和异常处理。
        若失败则抛出异常，由上层决定是否重试。
        """
        try:
            reasoning, content = await asyncio.wait_for(
                self.client.completion(),
                timeout=file_utils.COMPLETION_TIMEOUT
            )
            return reasoning, content
        except asyncio.TimeoutError:
            print(f"completion 超时 (超过 {file_utils.COMPLETION_TIMEOUT} 秒)")
            raise
        except Exception as e:
            print(f"completion 异常: {type(e).__name__}: {e}")
            raise

    async def run(self):
        """主运行循环"""
        if not await self._create_client():
            return

        if os.path.exists(file_utils.PAUSE_FLAG_FILE):
            os.remove(file_utils.PAUSE_FLAG_FILE)

        ongoing = True
        while ongoing:
            
            # paused 的逻辑是有意设计成这样的,请不要修改
            if self.paused:
                print("删除.pause以继续")
                while os.path.exists(file_utils.PAUSE_FLAG_FILE):
                    await asyncio.sleep(1)
                self.paused = False
            if os.path.exists(file_utils.PAUSE_FLAG_FILE):
                self.paused = True
            
            current_msg = file_utils.read_and_clear_message(file_utils.MESSAGE_FILE) or file_utils.initial_message(self.args)
            # 防止发送空消息导致循环卡死
            if not current_msg:
                current_msg = "Hello, please start the conversation."
            self.round_num += 1

            if not await self._ensure_connected():
                break

            print(f"\n{'='*50}\n第 {self.round_num} 轮：发送消息...")
            await self.client.send_prompt(
                f"user的第{self.round_num}轮输入\n{current_msg}\nuser的第{self.round_num}轮输入已结束"
            )

            reasoning, content = "", ""
            completion_success = False
            for attempt in range(file_utils.MAX_RECONNECT_ATTEMPTS + 1):
                try:
                    reasoning, content = await self._safe_completion()
                    completion_success = True
                    break
                except Exception:
                    if attempt == file_utils.MAX_RECONNECT_ATTEMPTS:
                        print("达到最大重试次数，退出")
                        ongoing = False
                        break
                    print(f"completion 失败，尝试重连并重试 ({attempt+1}/{file_utils.MAX_RECONNECT_ATTEMPTS})...")
                    if not await self._ensure_connected():
                        break
                    await self.client.send_prompt(
                        f"user的第{self.round_num}轮输入\n{current_msg}\nuser的第{self.round_num}轮输入已结束"
                    )
                    continue

            if not completion_success:
                break

            file_utils.save_response(content, file_utils.THIS_RESPONSE_FILE)
            file_utils.log_entry(self.round_num, current_msg, reasoning, content)

            print("=" * 30)
            print(f"推理过程:\n{reasoning}\n")
            print("=" * 30)
            print(f"回复内容:\n{content}\n")

            if file_utils.is_command_present(content):
                commands = file_utils.extract_commands(content)
                print("=" * 30)
                print(f"检测到 {len(commands)} 条命令，开始执行...")
                output = await CommandExecutor.execute_all(commands)
                print("=" * 30)
                print(f"工具输出汇总:\n{output}")

                with open(file_utils.LOG_FILE, "a", encoding="utf-8") as log:
                    log.write(f"\n{'='*25}\n工具输出\n{'='*25}\n")
                    log.write(output + "\n")

                with open(file_utils.MESSAGE_FILE, "w", encoding="utf-8") as f:
                    f.write(output)
            else:
                print("未检测到命令，返回系统提示。")
                output = file_utils.initial_message(self.args)
                with open(file_utils.MESSAGE_FILE, "w", encoding="utf-8") as f:
                    f.write(output)

            file_utils.save_response(content, file_utils.LAST_RESPONSE_FILE)

        print("=" * 30)
        print("Agent 结束")
        if self.client:
            await self.client.close()


async def main_async(args):
    """异步主函数，创建 Agent 并运行"""
    agent = Agent(args)
    await agent.run()


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