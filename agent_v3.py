# [START] AGENT-PKG
import asyncio
import argparse
import websockets
import sys

# [END] AGENT-PKG


from plugins import Plugin, LogPlugin
from plugins.prompt import DefaultPrompt


# [START] ADAPTERS-IMPORT
from adapters.base import MasterClient
from adapters.deepseek_webapp import DeepSeekWebApp

# [END] ADAPTERS-IMPORT


# [START] AGENT-CORE
class Agent:
    def __init__(self, client: MasterClient, plugins: dict[str, Plugin], args: dict):
        self.args = args
        self.client = client
        self.plugins = plugins

    # [START] AGENT-RUN
    async def run(self):
        listen_task = asyncio.create_task(self.client.listen())
        try:
            # 获取可用浏览器列表
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

            # 配对
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

            # 匹配目标 URL
            await self.client.call_match("chat.deepseek.com")

            # 主循环
            while True:
                req = {}

                # 前置插件处理（可能多次执行）
                should_loop = True
                while should_loop:
                    should_loop = False
                    for plugin in self.plugins.values():
                        # 修改：使用 .values() 直接迭代，避免重复字典查找
                        if plugin.before_prompt(self.args, req):
                            should_loop = True

                # 确定发送的文本和图片
                # 修改：优先使用 req 中的值，其次 args 中的 prompt，最后 args 中的 default_prompt
                prompt_text = req.get("prompt")
                prompt_image = req.get("image")

                # 若没有有效提示词，则退出循环
                if not prompt_text:
                    print("⚠️ 无有效提示词，退出对话。")
                    continue

                # 发送消息
                if prompt_image:
                    await self.client.call_send_prompt(
                        text=prompt_text, image_b64=prompt_image
                    )
                else:
                    await self.client.call_send_prompt(text=prompt_text)

                # 等待响应
                think, response = await self.client.pop_response()
                resp = {"think": think, "response": response}

                # 后置插件处理
                should_loop = True
                while should_loop:
                    should_loop = False
                    for plugin in self.plugins.values():
                        if plugin.after_prompt(self.args, req, resp):
                            should_loop = True

                # 退出条件：提示词为 exit/quit 或响应包含退出指令
                if self.args.get("should_exit"):
                    break

        except Exception as e:
            print(f"运行出错: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # 确保后台任务被取消
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
    # 修改：将 url 设为可选位置参数，使用 nargs='?' 并提供默认值
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

    # 构造传给 Agent 的参数字典
    agent_args = {
        "prompt": args.message,
        "new_chat": args.new_chat,
        "default_prompt": args.default_prompt,
        "next_image": None,  # 如果需要图片，可从命令行扩展
    }

    # 连接 WebSocket
    async with websockets.connect(args.url) as ws:
        client = DeepSeekWebApp(ws)
        # 插件列表：顺序很重要，DefaultPrompt 应在最后设置默认值
        plugins = {
            "logger": LogPlugin(),
            "default_prompt": DefaultPrompt(
                args.default_prompt
            ),  # 修改：DefaultPrompt 无参，从 args 读取
        }
        agent = Agent(client, plugins, agent_args)
        await agent.run()


# [END] AGENT-MAIN

if __name__ == "__main__":
    asyncio.run(main())
