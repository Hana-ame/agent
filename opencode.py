import sys
import subprocess
import json
from pathlib import Path
class Opencode:
    """OpenCode 交互封装类。
    提供 models_list.json 读取、agent 列表查询和 prompt 执行等功能。
    Args:
        models_list: 模型列表文件路径（相对于脚本所在目录），默认为 "models_list.json"。
    """
    # @opencode 这样改。
    def __init__(self, models_list="models_list.json"):
        self._models_path = Path(__file__).parent / models_list
    def _run_command(self, cmd, stream_stderr=True, stream_stdout=False):
        """执行命令，实时打印 stderr，捕获 stdout 并返回。
        
        返回: subprocess.CompletedProcess 类似的命名元组 (returncode, stdout, stderr)
        """
        print(f"[Debug] 执行命令: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,          # 行缓冲
        )
        stdout_lines = []
        stderr_lines = []
        # 为了同时读取 stdout 和 stderr 而不死锁，使用 communicate 或线程。
        # 简单起见，这里用 readlines + 轮询，但更好的做法是用线程。
        # 我们采用简易实现：只实时处理 stderr，stdout 等待进程结束后读取全部。
        # 但这样会失去 stdout 的实时性，通常 opencode 的结果都在最后，可以接受。
        # 如果你希望 stdout 也能实时看到，可以改为下面更完善的实现。
        
        # 更健壮的方式：使用线程分别读取 stdout 和 stderr
        import threading
        def read_stream(stream, is_stderr=False):
            """读取流，如果是 stderr 实时打印，同时存入列表"""
            for line in iter(stream.readline, ''):
                if is_stderr:
                    if stream_stderr:
                        sys.stderr.write(line)
                        sys.stderr.flush()
                    stderr_lines.append(line)
                else:                    
                    if stream_stdout:
                        sys.stderr.write(line)
                        sys.stderr.flush()
                    stdout_lines.append(line)
            stream.close()
        t1 = threading.Thread(target=read_stream, args=(process.stdout, False))
        t2 = threading.Thread(target=read_stream, args=(process.stderr, True))
        t1.start()
        t2.start()
        process.wait()
        t1.join()
        t2.join()
        stdout_str = ''.join(stdout_lines)
        stderr_str = ''.join(stderr_lines)
        print(f"[Debug] returncode: {process.returncode}")
        if stdout_str:
            print(f"[Debug] stdout: {stdout_str}")
        if stderr_str:
            print(f"[Debug] stderr: {stderr_str}")
        # 返回标准的 subprocess.CompletedProcess 对象
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout_str,
            stderr=stderr_str,
        )
    def list_models(self):
        """从 models_list.json 获取可用模型列表并返回格式化字符串。
        models_list.json 每行一个模型标识，该方法读取后以换行符拼接返回。
        """
        try:
            with open(self._models_path, "r", encoding="utf-8") as f:
                models = [line.strip() for line in f if line.strip()]
            if not models:
                return ""
            return "\n".join(models)
        except FileNotFoundError:
            raise RuntimeError(
                f"models_list.json not found at {self._models_path}"
            )
        except IOError as e:
            raise RuntimeError(f"Failed to read models_list.json: {e}")
    def list_agents(self):
        # opencode agent list
        result = self._run_command(["opencode", "agent", "list"])
        if result.returncode != 0:
            raise RuntimeError(f"list_agents failed: {result.stderr}")
        return result.stdout
    def run_prompt_json(self, prompt, agent="", model=""):
        cmd = ["opencode", "run"]
        if agent:
            cmd.extend(["--agent", agent])
        if model:
            cmd.extend(["--model", model])
        cmd.append(prompt)
        result = self._run_command(cmd)
        output = result.stdout.strip()
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"output": output, "success": result.returncode == 0}
    def run_prompt(self, prompt, agent="", model=""):
        """执行 opencode prompt，支持 @filename 语法自动读取文件内容。
        Args:
            prompt: 提示词文本，或以 @ 开头的文件路径（如 @opencode.py）
            agent: 指定 Agent 名称（可选）
            model: 指定模型名称（可选）
        Returns:
            str: opencode 执行结果 stdout
        """
        # 支持 @filename 语法：自动读取文件内容作为 prompt
        resolved_prompt = self._resolve_at_prompt(prompt)
        # opencode run --agent EditFile --model google/gemma-4-31b-it  "@opencode.py"
        cmd = ["opencode", "run"]
        if agent:
            cmd.extend(["--agent", agent])
        if model:
            cmd.extend(["--model", model])
        cmd.append(resolved_prompt)
        result = self._run_command(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"run_prompt failed: {result.stderr}")
        return result.stdout
    def _resolve_at_prompt(self, prompt):
        """解析 @filename 语法，返回实际 prompt 内容。
        如果 prompt 以 @ 开头，则读取对应文件内容作为 prompt；
        否则原样返回。
        """
        if not prompt or not prompt.startswith("@"):
            return prompt
        filepath = prompt[1:]  # 去掉 @ 前缀
        path = Path(filepath)
        # 如果是相对路径，基于脚本所在目录解析
        if not path.is_absolute():
            path = Path(__file__).parent / filepath
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise RuntimeError(f"Prompt file not found: {path}")
        except IOError as e:
            raise RuntimeError(f"Failed to read prompt file {path}: {e}")
def main():
    opencode = Opencode()
    result = opencode.run_prompt(
        "列举100个小猫的名字，和100个小狗的名字",
        agent="Null",
        # model="siliconflow-cn/THUDM/GLM-Z1-9B-0414",
        model="siliconflow-cn/Qwen/Qwen3-8B",
    )
    print(f"\n[Done] 结果:\n{result}")
if __name__ == "__main__":
    main()