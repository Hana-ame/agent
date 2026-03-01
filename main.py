import sys
import asyncio
from llm_client import LLMClient

def save_response(text: str) -> None:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines.pop(0)
    if lines and lines[-1].startswith("```"):
        lines.pop()
    with open("LAST_RESPONSE.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def read_and_clear_message() -> str:
    content = ""
    try:
        with open("MESSAGE.txt", "r", encoding="utf-8") as f:
            content = f.read()
        with open("MESSAGE.txt", "w", encoding="utf-8") as f:
            f.write("")
    except FileNotFoundError:
        pass
    return content

async def main():
    # 1. 初始化封装类
    client = LLMClient("wss://d.810114.xyz/ws/client")

    # 2. 连接并尝试配对 DeepSeek 浏览器
    success = await client.connect_and_pair("deepseek")
    if not success:
        print("❌ 配对失败！")
        return
    print("✅ 成功独占配对浏览器！")

    # 3. 发送最新信息
    print("✍️ 发送信息...")
    msg = ""
    if len(sys.argv) > 1:
        msg = sys.argv[1]
    if not msg:
        msg = read_and_clear_message()
    if not msg:
        try:
            with open("MESSAGE_DEFAULT.txt", "r", encoding="utf-8") as f:
                msg = f.read()
        except FileNotFoundError:
            pass
    
    await client.send_prompt(msg)

    # 4. 监听信息流并获取最终结果
    print("⏳ 等待 LLM 回复...\n")
    
    # 🌟 所有的 UI 实时更新和流处理都在 completion() 中执行
    reasoning, content = await client.completion()

    # 5. 打印与保存
    print("\n\n" + "=" * 40)
    print("🧠 REASONING (思考过程):")
    print(reasoning)
    print("-" * 40)
    print("📝 CONTENT (最终回答):")
    print(content)
    save_response(content)
    print("=" * 40)

if __name__ == "__main__":
    asyncio.run(main())