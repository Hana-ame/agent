import subprocess
import copy
from search import search
import json
import os
from completions import completions


class Agent:
    def __init__(self, path="./agents/test"):
        self.path = path
        # 预定义类型提示，建议给个默认值防止 json 里没有该字段时报错
        self.system_prompt: str = ""
        self.describe: str = ""
        self.notes: str = ""

        # -------------------------------------------------
        # 1. 读取 agent.json 并注入到 self (变成属性)
        # -------------------------------------------------
        agent_config_path = os.path.join(self.path, "agent.json")

        # 建议加个判断文件是否存在，防止报错
        if os.path.exists(agent_config_path):
            with open(agent_config_path, "r", encoding="utf-8") as f:
                # 核心逻辑：读取字典并更新到对象的 __dict__
                config_data = json.load(f)
                self.__dict__.update(config_data)
        else:
            print(f"[Warning] {agent_config_path} not found.")

        # -------------------------------------------------
        # 2. 读取 template.json 并存入 self.template
        # -------------------------------------------------
        self.template = {}
        template_config_path = os.path.join(self.path, "template.json")

        if os.path.exists(template_config_path):
            with open(template_config_path, "r", encoding="utf-8") as f:
                self.template = json.load(f)
        else:
            print(f"[Warning] {template_config_path} not found.")

    def completions(self, prompts: list[str | dict] = []):
        messages = [{"role": "system", "content": self.system_prompt}]
        for prompt in prompts:
            if isinstance(prompt, str):
                messages.append(
                    {
                        "role": "assistant",
                        "content": prompt,
                    }
                )  
            else:
                messages.append(prompt)

        payload = self.template.copy()
        payload["messages"] = messages
        return completions(payload=payload)

    def __str__(self):
        return f"system_prompt:{self.system_prompt}, describe:{self.describe}, notes:{self.notes}"
        pass


class ShellAgent(Agent):
    def __init__(self, path: str):
        super().__init__(path)

    def execute(self, cmd: str):
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        return result.stdout, result.stderr


class ExecuteAgent(Agent):
    def __init__(self, path: str):
        super().__init__(path)

    def execute(self, cmd: str):
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        return result.stdout, result.stderr


class SearchAgent(Agent):
    def __init__(self, path: str):
        super().__init__(path)

    def search(self, query: str):
        result = search(query=query)

        return result


class FileAgent(Agent):
    def __init__(self, path: str):
        super().__init__(path)

    def read(self, path: str):
        with open(path) as f:
            return f.read()

    def write(self, path: str, content: str):
        with open(path, "wt") as f:
            f.write(content)
