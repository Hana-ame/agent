import sys
import asyncio
import json
from llm_client import LLMClient


def save_response(text: str) -> None:
    """
    Save the given text to LAST_RESPONSE.txt.
    If the first line starts with ```, it is removed.
    If the last line starts with ```, it is removed.
    """
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines.pop(0)
    if lines and lines[-1].startswith("```"):
        lines.pop()
    with open("LAST_RESPONSE.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def read_and_clear_message() -> str:
    """
    Read the content of MESSAGE.txt, clear the file, and return the original content.
    If the file does not exist, return an empty string and do nothing.
    """
    content = ""
    try:
        with open("MESSAGE.txt", "r", encoding="utf-8") as f:
            content = f.read()
        # Clear the file by opening in write mode (truncates)
        with open("MESSAGE.txt", "w", encoding="utf-8") as f:
            f.write("")
    except FileNotFoundError:
        # If file doesn't exist, return empty string without creating it
        pass
    return content


async def main():
    # 1. 初始化封装类
    client = LLMClient()

    # 2. 连接并尝试配对 DeepSeek 浏览器
    print("⏳ 正在请求配对 DeepSeek 浏览器...")
    success = await client.connect_and_pair("deepseek")
    if not success:
        print("❌ 配对失败！请确保浏览器已打开 DeepSeek 页面并且注入了脚本。")
        return
    print("✅ 成功独占配对浏览器！")

    # # 3. 执行新建对话
    # print("✨ 执行新建对话...")
    # await client.new_chat()
    # await asyncio.sleep(1) # 给页面一点反应时间

    # 4. 发送最新信息
    print("✍️ 发送信息...")
    msg = sys.argv[1]
    if not msg:
        msg = read_and_clear_message()
    if not msg:
        print("msg 为空")
        return
    await client.send_prompt(msg)

    # 5. 监听信息流
    print("⏳ 等待 LLM 回复...\n")
    while True:
        raw_msg = await client.ws.recv()
        msg = json.loads(raw_msg)

        # 对方可能断开连接
        if msg.get("type") == "system" and msg.get("content") == "partner_disconnected":
            print("\n🔴 浏览器端已断开连接")
            break

        # ==========================================
        # 🌟 直接调用封装方法，得到两个字符串
        # ==========================================
        reasoning, content = client.completion(msg)

        if msg.get("type") == "token":
            # 实时进度展示
            print(
                f"\r[思考字符: {len(reasoning)}] | [回答字符: {len(content)}]",
                end="",
                flush=True,
            )

        if client.is_finished:
            print("\n\n" + "=" * 40)
            print("🧠 REASONING (思考过程):")
            print(reasoning)
            print("-" * 40)
            print("📝 CONTENT (最终回答):")
            print(content)
            save_response(content)
            print("=" * 40)
            break


if __name__ == "__main__":
    asyncio.run(main())
