import time
from abc import ABC, abstractmethod


class Plugin(ABC):
    @abstractmethod
    def before_prompt(self, args: dict, req: dict):
        pass

    @abstractmethod
    def after_prompt(self, args: dict, req: dict, resp: dict):
        pass


class LogPlugin(Plugin):
    def __init__(self):
        self.flag = False
        
    def before_prompt(self, args, req):
        print("[LOG] Sending prompt...")
        return False

    def after_prompt(self, args, req, resp):
        print(f"[LOG] Received response: {resp.get('response', '')[:50]}...")

        # 获取响应内容
        response_text = resp.get("response", "")
        # 分割成行，并获取最后一行
        lines = response_text.splitlines()
        if lines:
            last_line = lines[-1].strip()  # 去除可能的空白字符
        else:
            last_line = ""

        if self.flag:
            time.sleep(180)    

        # 如果最后一行恰好是 "★★★★★"
        if last_line == "★★★★★":
            self.flag = True
            return False
        else:
            self.flag = False
            return False
