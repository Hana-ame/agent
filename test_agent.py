# [START] AGENT-PKG
# version: 001
# 上下文：主控程序入口。先决调用：无。后续调用：AGENT-MAIN。
# 输入参数：无
# 输出参数：无
import sys
import asyncio
import os
import argparse
from pathlib import Path

from llm_client import LLMClient
from parser_rules import RuleProcessor
from file_utils import read_system_prompt, save_agent_content
#[END] AGENT-PKG

# [START] AGENT-CORE
# version: 001
# 上下文：系统的主体控制器。先决调用：参数解析完成。后续调用：建立连接并拉起循环。
# 输入参数：args (argparse.Namespace)
# 输出参数：无
class Agent:
    def __init__(self, args):
        self.args = args
        self.round_num = 0
        self.paused = False
        self.client: LLMClient = None
        self._reconnect_attempts = 0
        self._first_run = True

        self.root_path = os.getcwd()
        self.agent_dir = os.path.join(self.root_path, ".agent")
        os.makedirs(self.agent_dir, exist_ok=True)

        self.msg_file = os.path.join(self.agent_dir, "MESSAGE.txt")
        self.pause_file = os.path.join(".pause")
        self.log_file = os.path.join(self.agent_dir, "LOG.txt")

        self.last_response_file = os.path.join(self.agent_dir, "LAST_RESPONSE.txt")
        self.this_response_file = os.path.join(self.agent_dir, "THIS_RESPONSE.txt")

        self.utils_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils.py")
        self.processor = RuleProcessor(self.root_path, self.agent_dir, self.utils_path)


    # [START] AGENT-ENSURE-CONN
    # version: 001
    async def _ensure_connected(self) -> bool:
        if self.client and not self.client.is_finished:
            return True
        if self.client:
            await self.client.close()

        while self._reconnect_attempts < 5:
            self._reconnect_attempts += 1
            print(f"尝试重连 ({self._reconnect_attempts}/5)...")
            if await self._create_client():
                return True
            await asyncio.sleep(2 * self._reconnect_attempts)
        return False
    # [END] AGENT-ENSURE-CONN

    # [START] AGENT-POP-THOUGHT
    # version: 001
    # 上下文：从 THOUGHTS.md 中获取并移除第一条想法。
    # 输入参数：无
    # 输出参数：想法内容或 None
    def _pop_thought(self) -> str | None:
        thoughts_file = os.path.join(self.agent_dir, "THOUGHTS.md")
        if not os.path.exists(thoughts_file):
            return None
        try:
            with open(thoughts_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            new_lines = []
            thought = None
            for line in lines:
                stripped = line.lstrip()
                if thought is None and stripped.startswith('- '):
                    thought = stripped[2:].strip()
                    # 跳过该行，不加入 new_lines
                else:
                    new_lines.append(line)
            if thought:
                with open(thoughts_file, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                return thought
            else:
                return None
        except Exception:
            return None
    # [END] AGENT-POP-THOUGHT

    # [START] AGENT-RUN
    # version: 002  # 添加 thoughts 功能
    async def run(self):
        if not await self._create_client():
            return

        if os.path.exists(self.pause_file):
            os.remove(self.pause_file)

        ongoing = True
        while ongoing:

            if self.paused:
                print("删除 .agent/.pause 以继续")
                while os.path.exists(self.pause_file):
                    await asyncio.sleep(1)
                self.paused = False
            if os.path.exists(self.pause_file):
                self.paused = True

            current_msg = ""
            if os.path.exists(self.msg_file):
                with open(self.msg_file, 'r', encoding='utf-8') as f:
                    current_msg = f.read().strip()
                open(self.msg_file, 'w').close()

            if not current_msg and self.args.message:
                current_msg = self.args.message
            if not current_msg:
                # 尝试从想法中获取
                thought = self._pop_thought()
                if thought:
                    current_msg = thought
                    print(f"从 THOUGHTS.md 提取想法: {thought}")
                else:
                    current_msg = read_system_prompt()

            self.round_num += 1

            if not await self._ensure_connected():
                break

            print(f"\n{'='*50}\n第 {self.round_num} 轮：发送消息...")
            await self.client.send_prompt(f"user的第{self.round_num}轮输入\n\n{current_msg}\n\nuser的第{self.round_num}轮输入结束")

            reasoning, content = "", ""
            try:
                reasoning, content = await asyncio.wait_for(self.client.completion(), timeout=60*60)
            except Exception as e:
                print(f"获取 completion 失败: {e}")
                break

            print("=" * 30)
            print(f"推理过程:\n{reasoning}\n")
            print("=" * 30)
            print(f"回复内容:\n{content}\n")

            save_agent_content(self, content)

            processor_output = await self.processor.process(content)

            if processor_output:
                print("=" * 30)
                print(f"规则执行结果汇总:\n{processor_output}")

                with open(self.log_file, "a", encoding="utf-8") as log:
                    log.write(f"\n{'='*25}\n第 {self.round_num} 轮输出\n{'='*25}\n")
                    log.write(processor_output + "\n")

                with open(self.msg_file, "w", encoding="utf-8") as f:
                    f.write(processor_output)
            else:
                print("未匹配到任何操作规则。")
                # 如果没有生成下一轮消息，且对话仍在继续，下一轮循环将尝试想法或系统提示

        print("=" * 30)
        print("Agent 结束")
        if self.client:
            await self.client.close()
    # [END] AGENT-RUN
# [END] AGENT-CORE

# [START] AGENT-MAIN
# version: 001
def main():
    parser = argparse.ArgumentParser(description="Agent 客户端核心引擎")
    parser.add_argument("connection", default="wss://d.810114.xyz/ws/client", help="连接参数 (WS URL 或 profile name)")
    parser.add_argument("-m", "--message", help="直接提供消息内容启动")
    parser.add_argument("-p", "--payload", default="default.json", help="仅 HTTP 模式所需的 payload 模板")
    parser.add_argument("--new-chat", action="store_true", help="强制清除历史记忆")
    args = parser.parse_args()

    agent = Agent(args)
    asyncio.run(agent.run())

if __name__ == "__main__":
    main()
# [END] AGENT-MAIN