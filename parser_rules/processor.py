# [START] RULE-PROCESSOR
# version: 0.0.1
# 上下文：统筹所有激活的模式匹配规则。

import asyncio
from .base import BaseRule
from .command_rule import CommandRule
from .codeblock_rule import CodeBlockRule

class RuleProcessor:
    def __init__(self, root_path: str, agent_dir: str, utils_py_path: str):
        self.root_path = root_path
        self.rules : list[BaseRule] =[
            CommandRule(root_path, utils_py_path),
            CodeBlockRule(root_path, agent_dir)
        ]

    # [START] RULE-PROCESSOR-EXEC
    # version: 0.0.1
    # 上下文：接收模型单次回复文本，按注册顺序遍历执行所有规则。
    # 输入参数：text (str)
    # 输出参数：所有规则匹配动作产生的综合回执 (str)
    async def process(self, text: str) -> str:
        all_outputs =[]
        for rule in self.rules:
            out = await rule.match_and_execute(text)
            if out:
                all_outputs.append(out)
        return "\n---\n".join(all_outputs)
    # [END] RULE-PROCESSOR-EXEC
# [END] RULE-PROCESSOR