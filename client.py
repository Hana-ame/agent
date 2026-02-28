import asyncio
import json
import websockets

class DeepSeekStreamProcessor:
    def __init__(self):
        # State buffers to accumulate text
        self.reasoning_buffer = []
        self.content_buffer = []

    def process_message(self, new_msg):
        """
        Signature: (new_msg) -> reasoning_text, content_text
        Returns the FULL accumulated text for both parts every time it's called.
        """
        # Ensure we are looking at the 'content' part of the message
        data = new_msg.get("content", {})
        if not data:
            return "".join(self.reasoning_buffer), "".join(self.content_buffer)

        # 1. Handle Fragment/Initial Metadata (Deep JSON structure)
        # Structure: {"v": {"response": {"fragments": [{"content": "...", "type": "THINK"}]}}}
        v_field = data.get("v")
        if isinstance(v_field, dict):
            fragments = v_field.get("response", {}).get("fragments", [])
            for frag in fragments:
                text = frag.get("content", "")
                f_type = frag.get("type", "")
                if text and isinstance(text, str):
                    self._append_to_buffer(f_type, text)

        # 2. Handle Incremental Updates (APPEND mode)
        # Structure: {"o": "APPEND", "p": "response/fragments/0/content", "v": "..."}
        elif data.get("o") == "APPEND":
            path = data.get("p", "")
            val = data.get("v", "")
            
            if isinstance(val, str):
                # In DeepSeek: fragments/0 is usually Thinking, fragments/1 is Response
                if "fragments/0" in path:
                    self.reasoning_buffer.append(val)
                else:
                    self.content_buffer.append(val)

        # 3. Handle Simple Token Mode
        # Structure: {"v": "..."}
        elif isinstance(v_field, str):
            # If it's a raw string without a path, it's usually the final response
            self.content_buffer.append(v_field)

        return "".join(self.reasoning_buffer), "".join(self.content_buffer)

    def _append_to_buffer(self, f_type, text):
        if f_type == "THINK":
            self.reasoning_buffer.append(text)
        else:
            self.content_buffer.append(text)

async def run_client():
    url = "ws://localhost:8765/ws/client"
    # Instantiate the processor
    processor = DeepSeekStreamProcessor()
    
    try:
        async with websockets.connect(url) as ws:
            print("🟢 Connected to Golang Server")

            # 1. Match Site
            await ws.send(json.dumps({
                "type": "command", "command": "match", "params": {"target": "deepseek"}
            }))

            while True:
                raw_data = await ws.recv()
                msg = json.loads(raw_data)

                if msg.get("type") == "match_result" and msg["content"]["success"]:
                    print("✅ Match found. Sending prompt...")
                    await ws.send(json.dumps({
                        "type": "command",
                        "command": "send_prompt",
                        "params": {
                            "target": "deepseek",
                            "prompt": "im fool"
                        }
                    }))

                elif msg.get("type") == "token":
                    # Detect Completion
                    if "FINISHED" in str(msg):
                        break

                    # CALL THE FUNCTION with the required signature
                    reasoning, content = processor.process_message(msg)
                    
                    # Optional: Print progress dots so you know it's working
                    print(".", end="", flush=True)

            # FINAL OUTPUT using the accumulated strings from the processor
            final_reasoning, final_content = processor.process_message({})
            
            print("\n" + "="*50)
            print("REASONING (THINKING) CONTENTS:")
            print(final_reasoning.strip() if final_reasoning else "[No Reasoning Data]")
            print("-" * 50)
            print("RESPONSE CONTENTS:")
            print(final_content.strip())
            print("="*50)

    except Exception as e:
        print(f"\n❌ Python Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_client())