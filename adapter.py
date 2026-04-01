#!/usr/bin/env python3
"""
独立 OpenAI 适配器 - 带详细日志版本
每次请求新建连接，打印所有原始消息和状态变化。
"""

import asyncio
import json
import hashlib
import time
import traceback
from typing import Dict, Optional, Tuple
import websockets


class DeepSeekBridge:
    """管理 WebSocket 连接，解析 DeepSeek 响应"""

    def __init__(self, ws):
        self.ws = ws
        self.available_browsers = []
        self.paired = False

        # 响应解析状态
        self._is_think = False
        self._is_response = False
        self._is_finished = True
        self._think = []
        self._response = []
        self._inbox = asyncio.Queue()

        # 调试模式
        self.debug = True

    def log(self, msg):
        if self.debug:
            print(f"[Bridge] {msg}")

    async def send(self, channel: str, payload: dict):
        """发送消息到服务端"""
        msg = json.dumps({"channel": channel, "payload": payload}, ensure_ascii=False)
        self.log(f"发送: {msg}")
        await self.ws.send(msg)

    async def _set_status(self, status: str):
        """更新解析状态，当收到 FINISHED 时把完整响应放入队列"""
        prev_finished = self._is_finished
        self._is_finished = status == "FINISHED"
        self._is_think = status == "THINK"
        self._is_response = status == "RESPONSE"
        self.log(f"状态变更: {status} (finished:{self._is_finished})")

        if self._is_finished and not prev_finished:
            think_text = "".join(self._think)
            response_text = "".join(self._response)
            self.log(f"完成！think长度:{len(think_text)}, response长度:{len(response_text)}")
            await self._inbox.put((think_text, response_text))
            self._think, self._response = [], []

    async def _append_v(self, v):
        """追加内容到当前状态的缓冲区"""
        if not isinstance(v, str):
            return
        if v == "FINISHED":
            await self._set_status(v)
        elif self._is_think:
            self._think.append(v)
        elif self._is_response:
            self._response.append(v)
        # 忽略其他状态

    async def _deepseek_append_parser(self, obj: dict):
        """解析一个 fragment"""
        typ = obj.get("type")
        content = obj.get("content")
        self.log(f"fragment: type={typ}, content={content[:50] if content else ''}...")
        await self._set_status(typ)
        await self._append_v(content)

    async def _deepseek_object_parser(self, obj: dict):
        """解析一个完整的 deepseek 对象（可能包含多个片段）"""
        o = obj.get("o")
        p = obj.get("p")
        v = obj.get("v")

        self.log(f"object: o={o}, p={p}, v类型={type(v)}")

        # 普通对象或特定路径的对象
        if (o is None and p is None) or p in (
            "response/fragments/-1/content",
            "quasi_status",
            "response/status",
        ):
            if isinstance(v, dict):
                response = v.get("response")
                if response:
                    fragments = response.get("fragments")
                    if isinstance(fragments, list):
                        for frag in fragments:
                            await self._deepseek_append_parser(frag)
            await self._append_v(v)
        elif o == "APPEND":
            v_list = obj.get("v")
            if isinstance(v_list, list):
                for frag in v_list:
                    await self._deepseek_append_parser(frag)
            else:
                self.log(f"APPEND 但 v 不是列表: {v_list}")
        elif o == "BATCH":
            v_list = obj.get("v")
            if isinstance(v_list, list):
                for sub_obj in v_list:
                    await self._deepseek_object_parser(sub_obj)
            else:
                self.log(f"BATCH 但 v 不是列表: {v_list}")
        # 其他操作（如 SET）忽略

    async def listen(self):
        """后台监听 WebSocket 消息并解析"""
        try:
            async for raw in self.ws:
                self.log(f"原始消息: {raw[:200]}")
                data = json.loads(raw)
                channel = data.get("channel")
                payload = data.get("payload", {})

                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        pass

                if channel == "system":
                    command = payload.get("command")
                    if command == "list_result":
                        self.available_browsers = payload.get("message", [])
                        self.log(f"获取浏览器列表: {self.available_browsers}")
                    elif command == "pair_result":
                        if payload.get("message") == "success":
                            self.paired = True
                            self.log("✅ 配对成功")
                        else:
                            self.log(f"❌ 配对失败: {payload.get('message')}")
                else:
                    await self._deepseek_object_parser(payload)
        except websockets.exceptions.ConnectionClosed:
            self.log("🛑 连接断开")
        except Exception as e:
            traceback.print_exc()
            self.log(f"监听异常: {e}")

    async def pop_response(self, timeout: float = 60.0) -> Tuple[str, str]:
        """等待并返回响应（思考内容，最终回答），带超时"""
        try:
            think, response = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
            if think == "" and response == "":
                return await self.pop_response(timeout)
            return think, response
        except asyncio.TimeoutError:
            raise Exception(f"等待响应超时 {timeout} 秒")

    async def call_match(self, keyword: str):
        """导航到包含该关键词的域名"""
        await self.send("client", {"command": "match", "message": keyword})

    async def call_new_chat(self):
        """触发新建对话"""
        await self.send("client", {"command": "new_chat"})

    async def call_send_prompt(self, text: str, image_b64: Optional[str] = None):
        """发送文本（和可选图片）"""
        payload = {"command": "send_prompt", "message": text}
        if image_b64:
            payload["image"] = image_b64
        await self.send("client", payload)


class OpenAIAdapter:
    def __init__(self, ws_url: str = "wss://moonchan.publicvm.com/ws/client"):
        self.ws_url = ws_url

    async def _send_and_receive(self, text: str, new_chat: bool = False) -> str:
        """发送消息并等待响应"""
        print(f"\n=== 开始处理消息: {text[:50]}... ===")
        async with websockets.connect(self.ws_url) as ws:
            client = DeepSeekBridge(ws)
            listen_task = asyncio.create_task(client.listen())

            # 1. 请求浏览器列表
            await client.send("system", {"command": "list"})
            await asyncio.sleep(1.0)

            if not client.available_browsers:
                print("等待浏览器列表...")
                await asyncio.sleep(1.0)
                if not client.available_browsers:
                    raise Exception("没有可用的浏览器")

            # 2. 配对
            target = client.available_browsers[0]
            print(f"目标浏览器: {target}")
            await client.send(
                "system",
                {
                    "command": "pair",
                    "title": target.get("title"),
                    "type": target.get("type"),
                    "timestamp": target.get("timestamp"),
                },
            )
            await asyncio.sleep(1.0)
            if not client.paired:
                raise Exception("配对失败")

            # 3. 导航到 DeepSeek
            await client.call_match("chat.deepseek.com")
            await asyncio.sleep(1.0)

            # 4. 可选新建对话
            if new_chat:
                await client.call_new_chat()
                await asyncio.sleep(1.0)

            # 5. 发送消息
            await client.call_send_prompt(text=text)

            # 6. 等待响应
            think, response = await client.pop_response(timeout=120.0)

            # 清理
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass

            print(f"=== 收到响应，长度: {len(response)} ===")
            return response if response else think

    async def handle_request(self, request_data: Dict) -> Dict:
        """OpenAI API 格式的请求处理"""
        try:
            messages = request_data.get("messages", [])
            model = request_data.get("model", "deepseek-chat")

            # 提取最后一条用户消息
            user_msg = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                    break
            if not user_msg:
                raise ValueError("未找到用户消息")

            # 检查是否强制新建对话
            new_chat = any(
                msg.get("role") == "system"
                and msg.get("content", "").strip().upper() == "[NEW_CHAT]"
                for msg in messages
            )

            # 发送并获取响应
            answer = await self._send_and_receive(user_msg, new_chat)

            # 构建 OpenAI 响应
            return {
                "id": f"chatcmpl-{hashlib.md5(str(time.time()).encode()).hexdigest()[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_msg) // 4,
                    "completion_tokens": len(answer) // 4,
                    "total_tokens": (len(user_msg) + len(answer)) // 4,
                },
            }
        except Exception as e:
            traceback.print_exc()
            raise


# 测试入口
async def test():
    adapter = OpenAIAdapter()
    import sys

    msg = sys.argv[1] if len(sys.argv) > 1 else "Hello, how are you?"
    req = {"messages": [{"role": "user", "content": msg}]}
    try:
        resp = await adapter.handle_request(req)
        print(json.dumps(resp, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"测试失败: {e}")
    try:
        resp = await adapter.handle_request(req)
        print(json.dumps(resp, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(test())