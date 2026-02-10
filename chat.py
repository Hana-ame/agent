import os
import json
import sys
import argparse
from json_api_requester import JsonApiRequester
from json_payload_sender import JsonPayloadSender

# 颜色定义
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

class ChatCLI:
    def __init__(self, history_file: str = "history.json"):
        # 允许外部指定历史文件路径
        self.history_file = history_file
        self.current_reasoning = ""
        self.current_content = ""
        
        # 1. 初始化组件
        self.requester = JsonApiRequester(
            json_file_path="endpoint.json",
            sse_hook=self._handle_sse_chunk
        )
        self.sender = JsonPayloadSender(
            requester=self.requester,
            payload_file_path="payload.json"
        )
        
        # 初始化指定的历史记录文件
        if not os.path.exists(self.history_file):
            self._write_history([])
        else:
            # 确保 payload.json 里的 _context 指向当前指定的文件
            self.sender.update_payload_field("_context", self.history_file)

    def _handle_sse_chunk(self, chunk):
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if "reasoning_content" in delta and delta["reasoning_content"]:
            rc = delta["reasoning_content"]
            if not self.current_reasoning:
                print(f"{YELLOW}思考中...{RESET}")
            sys.stdout.write(f"{YELLOW}{rc}{RESET}")
            sys.stdout.flush()
            self.current_reasoning += rc
        elif "content" in delta and delta["content"]:
            c = delta["content"]
            if not self.current_content:
                print(f"\n\n{GREEN}回答：{RESET}")
            sys.stdout.write(c)
            sys.stdout.flush()
            self.current_content += c

    def _write_history(self, data):
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        # 关键：更新 payload.json 里的 _context 路径
        self.sender.update_payload_field("_context", self.history_file)

    def _read_history(self):
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return []

    def clear_history(self):
        self._write_history([])
        print(f"{BLUE}[系统] 已重置上下文文件: {self.history_file}{RESET}")

    def save_to_history(self, user_text, is_new=False):
        history = [] if is_new else self._read_history()
        history.append({"role": "user", "content": user_text})
        
        assistant_msg = {"role": "assistant", "content": self.current_content}
        if self.current_reasoning:
            assistant_msg["reasoning_content"] = self.current_reasoning
            
        history.append(assistant_msg)
        self._write_history(history)

    def execute_query(self, message: str, file_path: str = "", is_new: bool = False):
        if is_new:
            self.clear_history()

        full_query = ""
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                full_query += f"【参考文档：{file_path}】\n{f.read()}\n\n【参考文档：{file_path}结束】"
        
        full_query += message
        self.current_reasoning = ""
        self.current_content = ""

        # 调用 sender。如果 is_new，则不加载 context 文件
        self.sender.send_request_with_messages(
            dynamic_messages=[{"role": "user", "content": full_query}],
            use_context=(not is_new)
        )
        
        self.save_to_history(full_query, is_new=is_new)
        print("\n" + "-"*30)

    def interactive_loop(self):
        print(f"{BLUE}=== 对话模式 | Context: {self.history_file} | (exit 退出) ==={RESET}")
        while True:
            try:
                user_input = input(f"\n{BLUE}你: {RESET}").strip()
                if user_input.lower() in ['exit', 'quit']: break
                if not user_input: continue
                self.execute_query(user_input, is_new=False)
            except KeyboardInterrupt: break

def main():
    parser = argparse.ArgumentParser(description="DeepSeek CLI Chat 工具")
    parser.add_argument("-d", "--document", type=str, help="读取外部文档内容")
    parser.add_argument("-c", "--context", type=str, default="history.json", help="指定历史记录 JSON 文件 (默认: history.json)")
    parser.add_argument("--new", action="store_true", help="开启全新对话，清空当前指定的 context 文件")
    parser.add_argument("message", nargs="?", type=str, help="直接发送的消息内容")
    
    args = parser.parse_args()
    
    # 根据 -c 参数初始化 CLI
    chat = ChatCLI(history_file=args.context)

    if args.message or args.document:
        msg = args.message if args.message else "分析文档。"
        chat.execute_query(msg, args.document, is_new=args.new)
    else:
        if args.new:
            chat.clear_history()
        chat.interactive_loop()

if __name__ == "__main__":
    main()