from agent import *
from completions import get_content, user_prompt


class Task:
    def __init__(self, log_file="./log/logfile.txt", agents: list[Agent] = []):
        self.log_file = log_file
        self.count = 0
        self.agents_describe:str = ""
        a = agents[0]
        for agent in agents:
            content = get_content(a.completions([user_prompt("描述这个agent的作用"), user_prompt(str(agent))]) or {})
            self.agents_describe = self.agents_describe + content
        self.log(self.agents_describe)
            
    def log(self, content:str):
        with open(self.log_file, "wt+") as f:
            f.write(content)
            return 
        
        
if __name__== "__main__":
    a0 = Agent()
    a1 = Agent()
    Task(log_file="./log/testfile.txt", agents=[a0,a1])
