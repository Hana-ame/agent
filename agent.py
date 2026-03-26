import asyncio
import argparse
import websockets
import sys
import time
from plugins import Plugin, LogPlugin
from plugins.prompt import DefaultPrompt, SaveCodePlugin, RunBashCodeBlock
from adapters.base import MasterClient
from adapters.deepseek_webapp import DeepSeekWebApp
class Agent:
    def __init__(self, client: MasterClient, plugins: dict[str, Plugin], args: dict):
        self.args = args
        self.client = client
        self.plugins = plugins
    async def run(self):
        listen_task = asyncio.create_task(self.client.listen())
        try:
            await self.client.send("system", {"command": "list"})
            for i in range(10):
                await asyncio.sleep(1.0)  # 稍作等待给网络回传时间
                if not self.client.available_browsers:
                    print(
                        "⚠️ 当前没有可用的 Browser 注册在资源池中，请先让 Browser 上线。"
                    )
                    if i < 5:
                        continue
                    return
                else:
                    break
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
            for i in range(10):
                await asyncio.sleep(1.0)  # 稍作等待给网络回传时间
                if not self.client.paired:
                    if i < 5:
                        continue
                    print("⚠️ 无法完成配对，退出控制流。")
                    return
                else:
                    break
                
            await self.client.call_match("chat.deepseek.com")
            
            while True:
                req = {"prompt": ""}
                should_loop = True
                while should_loop:
                    should_loop = False
                    for plugin in self.plugins.values():
                        if plugin.before_prompt(self.args, req):
                            should_loop = True
                    if should_loop:
                        print('?')
                        time.sleep(5)
                prompt_text = req.get("prompt")
                prompt_image = req.get("image")
                if not prompt_text:
                    print("⚠️ 无有效提示词，退出对话。")
                    continue
                if prompt_image:
                    await self.client.call_send_prompt(
                        text=prompt_text, image_b64=prompt_image
                    )
                else:
                    await self.client.call_send_prompt(text=prompt_text)
                think, response = await self.client.pop_response()
                resp = {"think": think, "response": response}
                should_loop = True
                while should_loop:
                    should_loop = False
                    for plugin in self.plugins.values():
                        if plugin.after_prompt(self.args, req, resp):
                            should_loop = True
                    if should_loop:
                        print('?')
                        time.sleep(5)
                if self.args.get("should_exit"):
                    break
        except Exception as e:
            print(f"运行出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass
async def main():
    parser = argparse.ArgumentParser(description="Agent 客户端核心引擎")
    parser.add_argument(
        "url", nargs="?", default="ws://127.26.3.1:8080/ws/client", help="WebSocket URL"
    )
    parser.add_argument("-m", "--message", help="直接提供消息内容启动")
    parser.add_argument(
        "-p",
        "--payload",
        default="default.json",
        help="仅 HTTP 模式所需的 payload 模板",
    )
    parser.add_argument("--new-chat", action="store_true", help="强制清除历史记忆")
    parser.add_argument("--default-prompt", default="咕咕嘎嘎。", help="默认提示词")
    args = parser.parse_args()
    agent_args = {
        "prompt": args.message,
        "new_chat": args.new_chat,
        "default_prompt": args.default_prompt,
        "next_image": None,  # 如果需要图片，可从命令行扩展
    }
    async with websockets.connect(args.url) as ws:
        client = DeepSeekWebApp(ws)
        plugins = {
            "save_code":SaveCodePlugin(),
            "run_code": RunBashCodeBlock(),
            "default_prompt": DefaultPrompt(
                "请继续完成最开始的任务。或者接着探索其他解法。",
                ".agent/SYSTEM_PROMPT.txt"
            ),  # 修改：DefaultPrompt 无参，从 args 读取
            "log": LogPlugin(),
        }
        agent = Agent(client, plugins, agent_args)
        await agent.run()
if __name__ == "__main__":
    asyncio.run(main())
