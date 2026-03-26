from base import Plugin

class DefaultPrompt(Plugin):
    def __init__(self, default_prompt:str):
        self.default_prompt = default_prompt
        
    def before_prompt(self, args, req):
        if str(req.get("prompt", ""))  == "":
            req["prompt"] = self.default_prompt
        return False            

    def after_prompt(self, args, req, resp):
        return False