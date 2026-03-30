import sys
import traceback
import json
import asyncio
import websockets
from collections import deque
from .base import MasterClient
class DeepSeekWebApp(MasterClient):
    def __init__(self, ws):
        super().__init__(ws)
        self._is_think = False
        self._is_response = False
        self._is_finished = True
        self._think = []
        self._response = []
        self._inbox = asyncio.Queue()
    async def _set_status(self, type):
        prev_status_is_finished = self._is_finished
        self._is_finished = type == "FINISHED"
        self._is_think = type == "THINK"
        self._is_response = type == "RESPONSE"
        if self._is_finished and not prev_status_is_finished:
            await self._inbox.put(("".join(self._think), "".join(self._response)))
            self._think, self._response = [], []
    async def _append_v(self, v):
        if not isinstance(v, str):
            return
        if v == "FINISHED":
            await self._set_status(v)
        elif self._is_think:
            self._think.append(v)
        elif self._is_response:
            self._response.append(v)
    async def _deepseek_append_parser(self, obj: dict):
        type, content = obj.get("type"), obj.get("content")
        await self._set_status(type)
        await self._append_v(content)
    async def _deepseek_object_parser(self, obj: dict):
        o, p = obj.get("o"), obj.get("p")
        if (
            (o is None and p is None)
            or p == "response/fragments/-1/content"
            or p == "quasi_status"
            or p == "response/status"
        ):
            v = obj.get("v")
            if isinstance(v, dict):
                response = v.get("response")
                if response:
                    fragments = response.get("fragments")
                    if isinstance(fragments, list):
                        for v in fragments:
                            await self._deepseek_append_parser(v)
            await self._append_v(v)
        elif o == "APPEND":
            v_list = obj.get("v")
            if isinstance(v_list, list):
                for v in v_list:
                    await self._deepseek_append_parser(v)
            else:
                raise Exception(f"supposed list {v_list}")
        elif o == "BATCH":
            v_list = obj.get("v")
            if isinstance(v_list, list):
                for v in v_list:
                    await self._deepseek_object_parser(v)
            else:
                raise Exception(f"supposed list {v_list}")
        else:  # SET, etc.
            pass
    async def listen(self):
        try:
            async for raw_message in self.ws:
                data = json.loads(raw_message)
                channel = data.get("channel")
                payload: dict = data.get("payload", {})
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        pass
                if channel == "system":
                    print(f"\n<<< [收到 Object 回传] 原始报文: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    command = payload.get("command")
                    if command == "list_result":
                        self.available_browsers = payload.get("message", [])
                    elif command == "pair_result":
                        if payload.get("message") == "success":
                            self.paired = True
                            print("✅ [系统] 成功与 Browser 建立专属配对！")
                        else:
                            print(f"❌ [系统] 配对失败: {payload.get('message')}")
                else:
                    await self._deepseek_object_parser(payload)
        except websockets.exceptions.ConnectionClosed:
            print("🛑 连接已断开")
        except Exception as e:
            traceback.print_exc()
            stack_str = traceback.format_exc()
            print(f"❌ 监听发生异常: {stack_str}")
    async def pop_response(self):
        try:
            think, response = await self._inbox.get()
            if think == "" and response == "":
                return self.pop_response()
            else:
                return think, response
        except:
            return "error", "error"
async def run_test(ws_url):
    """
    负责连接 WebSocket、实例化 Client 并执行自动化下发步骤
    """
    async with websockets.connect(ws_url) as ws:
        client = DeepSeekWebApp(ws)
        print(f"🚀 控制端 (Client) 已连接到 Bridge Server: {ws_url}")
        listen_task = asyncio.create_task(client.listen())
        print("\n🔍 请求当前在线的 Browser 列表...")
        await client.send("system", {"command": "list"})
        await asyncio.sleep(1.0)  # 稍作等待给网络回传时间
        if not client.available_browsers:
            print("⚠️ 当前没有可用的 Browser 注册在资源池中，请先让 Browser 上线。")
            listen_task.cancel()
            return
        target_browser = client.available_browsers[0]
        print(f"\n🔗 尝试与选定的 Browser 建立配对...")
        await client.send(
            "system",
            {
                "command": "pair",
                "title": target_browser.get("title"),
                "type": target_browser.get("type"),
                "timestamp": target_browser.get("timestamp"),
            },
        )
        await asyncio.sleep(1.0)  # 等待配对确认
        if not client.paired:
            print("⚠️ 无法完成配对，退出控制流。")
            listen_task.cancel()
            return
        print("\n================ 开始执行 Browser 控制流 ================")
        print("\n[步骤 1] 校验浏览器当前域名是否匹配...")
        await client.call_match("chat.deepseek.com")
        print("\n[步骤 2] 触发网页 [新建对话] 按钮...")
        print("\n[步骤 3] 填充文本和图片并 [发送 Prompt]...")
        await client.call_send_prompt(
            text=sys.argv[1],
        )
        print(client._inbox)
        think, response = await client.pop_response()
        print(think)
        print(response)
        print("\n🎉 所有自动控制流下发完毕！保持在线以接收回传...")
        print(print("队列大小:", client._inbox.qsize()))
        try:
            print(await client.pop_response())
        except Exception as e:
            print(f"异常: {e}")
            import traceback
            traceback.print_exc()
        await listen_task
if __name__ == "__main__":
    WS_CLIENT_ENDPOINT = "ws://127.26.3.1:8080/ws/client"
    try:
        asyncio.run(run_test(WS_CLIENT_ENDPOINT))
    except KeyboardInterrupt:
        print("\n🛑 控制端被手动终止")
    except Exception as e:
        print(f"\n❌ 连接或运行发生错误: {e}")
