import json
import asyncio
import websockets

class MasterClient:
    def __init__(self, ws):
        """
        初始化客户端，传入已经建立连接的 websocket 对象
        """
        self.ws = ws
        self.available_browsers =[]
        self.paired = False

    async def send(self, channel, payload):
        """
        统一封装发送格式，满足 Go 服务端 Message 结构要求
        type Message struct { Channel string; Payload json.RawMessage }
        """
        message_obj = {
            "channel": channel,
            "payload": payload
        }
        msg_json = json.dumps(message_obj, ensure_ascii=False)
        print(f"\n>>> [向服务端发送] 包装数据: {msg_json}")
        if self.ws:
            await self.ws.send(msg_json)

    async def listen(self):
        """
        后台常驻协程：监听 WebSocket 收到的所有消息
        """
        try:
            async for raw_message in self.ws:
                # 1. 对于收到的 object，一律 print
                data = json.loads(raw_message)
                print(f"\n<<< [收到 Object 回传] 原始报文: {json.dumps(data, indent=2, ensure_ascii=False)}")

                channel = data.get("channel")
                payload = data.get("payload", {})

                # 如果 Go 服务端把 payload 当作字符串下发，进行二次解析
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        pass

                # 处理系统指令（保存列表、确认配对状态）
                if channel == "system":
                    command = payload.get("command")
                    if command == "list_result":
                        self.available_browsers = payload.get("message",[])
                    elif command == "pair_result":
                        if payload.get("message") == "success":
                            self.paired = True
                            print("✅ [系统] 成功与 Browser 建立专属配对！")
                        else:
                            print(f"❌ [系统] 配对失败: {payload.get('message')}")
                            
        except websockets.exceptions.ConnectionClosed:
            print("🛑 连接已断开")
        except Exception as e:
            print(f"❌ 监听发生异常: {e}")

    # ==========================================
    # 远程控制 JS Adapter 的具体方法
    # 发送通道须为 "client"
    # ==========================================
    async def call_match(self, domain_keyword):
        await self.send("client", {
            "command": "match",
            "message": domain_keyword
        })

    async def call_new_chat(self):
        await self.send("client", {
            "command": "new_chat"
        })

    async def call_send_prompt(self, text, image_b64=None):
        payload = {
            "command": "send_prompt",
            "message": text
        }
        if image_b64:
            payload["image"] = image_b64
        await self.send("client", payload)

    async def call_remove_msg(self):
        await self.send("client", {
            "command": "remove_msg"
        })
    
    async def pop_response(self):
        pass

# ==========================================
# 核心测试业务流 (提至外部)
# ==========================================
async def run_test(ws_url):
    """
    负责连接 WebSocket、实例化 Client 并执行自动化下发步骤
    """
    async with websockets.connect(ws_url) as ws:
        client = MasterClient(ws)
        print(f"🚀 控制端 (Client) 已连接到 Bridge Server: {ws_url}")

        # 启动并发后台监听任务
        listen_task = asyncio.create_task(client.listen())

        # 1. 握手与发现：查询当前连接到服务器的 Browser
        print("\n🔍 请求当前在线的 Browser 列表...")
        await client.send("system", {"command": "list"})
        await asyncio.sleep(1.0)  # 稍作等待给网络回传时间

        if not client.available_browsers:
            print("⚠️ 当前没有可用的 Browser 注册在资源池中，请先让 Browser 上线。")
            listen_task.cancel()
            return

        # 2. 锁定与配对：选择第一个 Browser 尝试绑定
        target_browser = client.available_browsers[0]
        print(f"\n🔗 尝试与选定的 Browser 建立配对...")
        await client.send("system", {
            "command": "pair",
            "title": target_browser.get("title"),
            "type": target_browser.get("type"),
            "timestamp": target_browser.get("timestamp")
        })
        await asyncio.sleep(1.0)  # 等待配对确认

        if not client.paired:
            print("⚠️ 无法完成配对，退出控制流。")
            listen_task.cancel()
            return

        # 3. 开始调用方法：控制前端的 JS 做事情
        print("\n================ 开始执行 Browser 控制流 ================")

        print("\n[步骤 1] 校验浏览器当前域名是否匹配...")
        await client.call_match("chat.deepseek.com")
        await asyncio.sleep(1.0)

        print("\n[步骤 2] 触发网页 [新建对话] 按钮...")
        await client.call_new_chat()
        await asyncio.sleep(1.5)

        print("\n[步骤 3] 填充文本和图片并 [发送 Prompt]...")
        # 这是一个透明 1x1 像素 PNG 的 Base64 示例
        # dummy_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        await client.call_send_prompt(
            text="你搜索今日天气，并直接回报我当前时间点气温。不要思考，也不要回答任何格式信息", 
            # image_b64=dummy_image_b64
        )
        await asyncio.sleep(30)  # 模拟等待 AI 思考及 DOM 渲染

        print("\n[步骤 4] 触发网页 [删除历史记录] 按钮...")
        # await client.call_remove_msg()

        print("\n🎉 所有自动控制流下发完毕！保持在线以接收回传...")
        
        # 保持连接阻塞，以持续接收后续可能来自浏览器的通知 (例如 match_result)
        await listen_task


if __name__ == "__main__":
    # 指向 Go 服务端分配给 client 角色的端点
    WS_CLIENT_ENDPOINT = "ws://127.26.3.1:8080/ws/client"
    
    try:
        # 直接在此处触发独立于 Class 之外的测试逻辑
        asyncio.run(run_test(WS_CLIENT_ENDPOINT))
    except KeyboardInterrupt:
        print("\n🛑 控制端被手动终止")
    except Exception as e:
        print(f"\n❌ 连接或运行发生错误: {e}")