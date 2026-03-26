from abc import ABC, abstractmethod
class Plugin(ABC):
    @abstractmethod
    def before_prompt(self, args, req):
        pass
    @abstractmethod
    def after_prompt(self, args, req, resp):
        pass
class LogPlugin(Plugin):
    def before_prompt(self, args, req):
        print("[LOG] Sending prompt...")
        return False
    def after_prompt(self, args, req, resp):
        print(f"[LOG] Received response: {resp.get('response', '')[:50]}...")
        return False