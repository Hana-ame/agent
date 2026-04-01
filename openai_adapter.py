#!/usr/bin/env python3
"""
OpenAI Adapter for DeepSeek Web UI - Debug Version
完全复制 run_test 的流程，并打印详细日志。
"""

import asyncio
import hashlib
import json
import sys
import time
import traceback
from typing import Dict

import websockets
from adapters.deepseek_webapp import DeepSeekWebApp  # 确保此导入可用


class OpenAIAdapter:
    def __init__(self, ws_url: str = "wss://moonchan.publicvm.com/ws/client", debug: bool = True):
        self.ws_url = ws_url
        self.debug = debug

    def log(self, msg):
        if self.debug:
            print(f"[DEBUG] {msg}")

    async def _send_and_receive(self, text: str, new_chat: bool = False) -> str:
        """完全复用 run_test 的流程，返回响应字符串"""
        self.log(f"开始处理消息: {text[:100]}...")
        async with websockets.connect(self.ws_url) as ws:
            client = DeepSeekWebApp(ws)
            listen_task = asyncio.create_task(client.listen())
            self.log("WebSocket 已连接，监听任务已启动")

            # 1. 获取浏览器列表
            await client.send("system", {"command": "list"})
            self.log("已发送 list 命令")
            await asyncio.sleep(1.0)

            if not client.available_browsers:
                self.log("未立即获取到浏览器，等待1秒...")
                await asyncio.sleep(1.0)
                if not client.available_browsers:
                    raise Exception("没有可用的浏览器")
            self.log(f"获取到浏览器列表: {client.available_browsers}")

            # 2. 配对
            target = client.available_browsers[0]
            await client.send(
                "system",
                {
                    "command": "pair",
                    "title": target.get("title"),
                    "type": target.get("type"),
                    "timestamp": target.get("timestamp"),
                },
            )
            self.log("已发送配对请求")
            await asyncio.sleep(1.0)

            if not client.paired:
                raise Exception("配对失败")
            self.log("配对成功")

            # 3. 导航到 DeepSeek 页面
            await client.call_match("chat.deepseek.com")
            self.log("已导航到 chat.deepseek.com")
            await asyncio.sleep(1.0)

            # 4. 可选新建对话
            if new_chat:
                await client.call_new_chat()
                self.log("已执行新建对话")
                await asyncio.sleep(1.0)

            # 5. 发送消息
            await client.call_send_prompt(text=text)
            self.log(f"已发送消息: {text[:50]}...")

            # 6. 等待响应
            self.log("等待响应中...")
            think, response = await client.pop_response()
            self.log(f"收到响应 - think长度:{len(think)} response长度:{len(response)}")

            # 清理
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass

            return response if response else think

    async def handle_request(self, request_data: Dict) -> Dict:
        """OpenAI 兼容接口"""
        try:
            self.log("收到请求: " + json.dumps(request_data, ensure_ascii=False)[:200])
            messages = request_data.get("messages", [])
            model = request_data.get("model", "deepseek-chat")

            # 提取最后一条用户消息
            user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            if not user_message:
                raise ValueError("消息中未找到用户消息")
            self.log(f"提取的用户消息: {user_message[:100]}...")

            # 检查是否强制新建对话
            new_chat = any(
                msg.get("role") == "system" and msg.get("content", "").strip().upper() == "[NEW_CHAT]"
                for msg in messages
            )
            if new_chat:
                self.log("检测到新建对话标记")

            response_text = await self._send_and_receive(user_message, new_chat)

            # 构建 OpenAI 格式响应
            return {
                "id": f"chatcmpl-{hashlib.md5(str(time.time()).encode()).hexdigest()[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message) // 4,
                    "completion_tokens": len(response_text) // 4,
                    "total_tokens": (len(user_message) + len(response_text)) // 4
                }
            }
        except Exception as e:
            traceback.print_exc()
            raise


async def test():
    adapter = OpenAIAdapter(debug=True)
    # 支持从命令行传入消息，否则使用默认
    msg = sys.argv[1] if len(sys.argv) > 1 else "Hello, how are you?"
    req = {"messages": [{"role": "user", "content": msg}]}
    try:
        resp = await adapter.handle_request(req)
        print(json.dumps(resp, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(test())