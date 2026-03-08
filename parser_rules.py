# [START] PARSER-RULES-PKG
# version: 001
# 上下文：模块装载起点。先决调用：无。后续调用：被 agent.py 导入实例化。
# 输入参数：无
# 输出参数：无
import re
import os
import sys
import hashlib
import asyncio
# [END] PARSER-RULES-PKG

# [START] BASE-RULE
# version: 001
# 上下文：定义匹配规则的抽象基类。先决调用：无。后续调用：具体规则类继承并实现核心方法。
# 输入参数：无
# 输出参数：无
class BaseRule:
    
    # [START] BASE-RULE-MATCH
    # version: 001
    # 上下文：由规则处理器分发文本时调用。先决调用：大模型完整输出 content。后续调用：返回处理结果合并入系统。
    # 输入参数：text (str)
    # 输出参数：执行结果汇总 (str)
    async def match_and_execute(self, text: str) -> str:
        raise NotImplementedError
    # [END] BASE-RULE-MATCH
# [END] BASE-RULE

# [START] COMMAND-RULE
# version: 001
# 上下文：在规则处理器中注册，用于拦截按行分割的单行终端指令。先决调用：BaseRule 接口规范。后续调用：通过 asyncio.create_subprocess_exec 执行子进程。
# 输入参数：root_path (str), utils_py_path (str)
# 输出参数：无
class CommandRule(BaseRule):
    def __init__(self, root_path: str, utils_py_path: str):
        self.root_path = root_path
        self.utils_py_path = utils_py_path

    # [START] COMMAND-RULE-MATCH
    # version: 001
    # 上下文：逐行扫描大模型输出的文本。先决调用：系统接收到 content。后续调用：拉起 sys.executable 执行 utils.py。
    # 输入参数：text (str)
    # 输出参数：执行结果或标准错误 (str)
    async def match_and_execute(self, text: str) -> str:
        results =[]
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("py utils.py"):
                parts = line.split(" ", 2)
                args = parts[2].split() if len(parts) > 2 else []
                
                cmd =[sys.executable, self.utils_py_path] + args
                
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=self.root_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                
                out_str = stdout.decode('utf-8', errors='replace').strip()
                err_str = stderr.decode('utf-8', errors='replace').strip()
                
                res = f"执行命令 [{line}]:\n"
                if out_str: res += f"{out_str}\n"
                if err_str: res += f"错误:\n{err_str}\n"
                if not out_str and not err_str: res += "执行成功 (无输出)\n"
                
                results.append(res)
                
        return "\n".join(results)
    # [END] COMMAND-RULE-MATCH
# [END] COMMAND-RULE

# [START] CODEBLOCK-RULE
# version: 001
# 上下文：在规则处理器中注册，用于拦截并提取 Markdown 代码块。先决调用：BaseRule 接口规范。后续调用：将提取的代码写入 .agent/ 目录下的哈希文件。
# 输入参数：agent_dir (str)
# 输出参数：无
class CodeBlockRule(BaseRule):
    def __init__(self, agent_dir: str):
        self.agent_dir = agent_dir

    # [START] CODEBLOCK-RULE-MATCH
    # version: 001
    # 上下文：正则扫描全文中带有 ``` 包裹的代码段。先决调用：系统接收到 content。后续调用：文件系统 I/O 操作。
    # 输入参数：text (str)
    # 输出参数：文件保存位置信息的通知文本 (str)
    async def match_and_execute(self, text: str) -> str:
        results = []
        pattern = re.compile(r'```([a-zA-Z0-9_\-+]+)?\n(.*?)```', re.DOTALL)
        
        for match in pattern.finditer(text):
            ext = match.group(1) or 'txt'
            content = match.group(2)
            
            hash_val = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
            filename = f"{hash_val}.{ext}"
            filepath = os.path.join(self.agent_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
                
            results.append(f"检测到代码块，已提取并保存至: .agent/{filename}")
            
        return "\n".join(results)
    # [END] CODEBLOCK-RULE-MATCH
# [END] CODEBLOCK-RULE

# [START] RULE-PROCESSOR
# version: 001
# 上下文：Agent 核心生命周期组件，统筹所有激活的模式匹配规则。先决调用：无。后续调用：在 agent.run 循环中被调用。
# 输入参数：root_path (str), agent_dir (str), utils_py_path (str)
# 输出参数：无
class RuleProcessor:
    def __init__(self, root_path: str, agent_dir: str, utils_py_path: str):
        self.rules =[
            CommandRule(root_path, utils_py_path),
            CodeBlockRule(agent_dir)
        ]

    # [START] RULE-PROCESSOR-EXEC
    # version: 001
    # 上下文：接收模型单次回复文本，按注册顺序遍历执行所有规则。先决调用：完成大模型推理。后续调用：将汇总的结果写回 MESSAGE.txt 闭环。
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
#[END] RULE-PROCESSOR