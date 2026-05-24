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

    def _run_command(self, cmd):
        # 使用 subprocess.run 替代 os.system，可以捕获 stdout/stderr 并返回结果
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        return result

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
    """主函数：运行 opencode run --agent EditFile "@database.py" 并打印结果。"""
    opencode = Opencode()
    result = opencode.run_prompt("@database.py", agent="EditFile")
    print(result)


if __name__ == "__main__":
    main()