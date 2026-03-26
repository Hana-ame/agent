# [START] AGENT-PKG
import asyncio
import argparse
import websockets

# [END] AGENT-PKG
from adapters.base import MasterClient
from adapters.deepseek_webapp import DeepSeekWebApp

from plugins import Plugin, LogPlugin
from plugins.prompt import DefaultPrompt


# [START] AGENT-CORE
class Agent:
    def __init__(self, client: MasterClient, plugins: dict[str, Plugin], args: dict):
        self.args = args
        self.client = client
        self.plugins = plugins

    # [START] AGENT-RUN
    async def run(self):

        try:
            listen_task = asyncio.create_task(self.client.listen())

            await self.client.send("system", {"command": "list"})

            if not self.client.available_browsers:
                print("⚠️ 当前没有可用的 Browser 注册在资源池中，请先让 Browser 上线。")
                listen_task.cancel()
                return
            # 2. 锁定与配对：选择第一个 Browser 尝试绑定
            target_browser = self.client.available_browsers[0]
            print(f"\n🔗 尝试与选定的 Browser 建立配对...")
            await self.client.send(
                "system",
                {
                    "command": "pair",
                    "title": target_browser.get("title"),
                    "type": target_browser.get("type"),
                    "timestamp": target_browser.get("timestamp"),
                },
            )
            if not self.client.paired:
                print("⚠️ 无法完成配对，退出控制流。")
                listen_task.cancel()
                return

            await self.client.call_match("chat.deepseek.com")

            while True:
                req = {}

                should_loop = True
                while should_loop:
                    should_loop = False
                    for k in self.plugins:
                        should_loop = should_loop or self.plugins[k].before_prompt(
                            self.args, req
                        )

                if req.get("prompt") and req.get("image"):
                    await self.client.call_send_prompt(
                        text=req["prompt"], image_b64=req["image"]
                    )
                elif req.get("prompt",""):
                    await self.client.call_send_prompt(text=req["prompt"])
                else:
                    await self.client.call_send_prompt(text=req["prompt"])

                think, response = await self.client.pop_response()
                resp = {"think": think, "response": response}

                should_loop = True
                while should_loop:
                    should_loop = False
                    for k in self.plugins:
                        should_loop = should_loop or self.plugins[k].after_prompt(
                            self.args, req, resp
                        )
        finally:
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass

    # [END] AGENT-RUN


# [END] AGENT-CORE


# [START] AGENT-MAIN
async def main():
    parser = argparse.ArgumentParser(description="Agent 客户端核心引擎")
    parser.add_argument(
        "url",
        default="wss://d.810114.xyz/ws/client",
        help="连接参数 (WS URL 或 profile name)",
    )
    parser.add_argument("-m", "--message", help="直接提供消息内容启动")
    parser.add_argument(
        "-p",
        "--payload",
        default="default.json",
        help="仅 HTTP 模式所需的 payload 模板",
    )
    parser.add_argument("--new-chat", action="store_true", help="强制清除历史记忆")

    args = parser.parse_args()

    async with websockets.connect(args.url) as ws:
        client = DeepSeekWebApp(ws)
        agent = Agent(
            client,
            {"logger": LogPlugin(), "default_prompt": DefaultPrompt("咕咕嘎嘎。")},
            {
                "prompt": args.message,
                "new_chat": args.new_chat,
            },
        )
        await agent.run()


if __name__ == "__main__":
    asyncio.run(main)
# [END] AGENT-MAIN
