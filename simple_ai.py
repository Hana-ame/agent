import requests
import json
import time, datetime
import os, sys
import gzip


class SimpleAI:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        default_model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        system_prompt: str = "",
        bug = False,
        **other_settings,
    ):
        """
        初始化客户端，在此处设置全局通用的参数。

        :param api_key: API Key
        :param base_url: API 基础地址
        :param default_model: 默认模型
        :param temperature: (全局设置) 随机性，0-2之间。越低越严谨，越高越发散。
        :param max_tokens: (全局设置) 最大回复长度
        :param top_p: (全局设置) 核采样概率
        :param other_settings: 其他想要全局携带的参数，如 frequency_penalty 等
        """
        Bearer = "Barer" if bug else "Bearer"
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

        # 将这些配置保存为默认参数字典
        self.default_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **other_settings,  # 允许传入其他任意兼容参数
        }

        self.headers = {
            "Authorization": f"{Bearer} {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, user_text: str, system_prompt: str = "", **kwargs) -> str:
        """
        发送请求。
        优先级逻辑：kwargs(本次调用传入) > self.default_params(类初始化设置)
        """
        url = self.base_url

        # 1. 准备消息历史
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})

        # 2. 构建 Payload (载荷)
        # 先加载类的默认设置 (temperature, max_tokens等)
        payload = self.default_params.copy()

        # 强制设置 model 和 messages
        payload["model"] = self.default_model
        payload["messages"] = messages
        payload["stream"] = False

        # 3. 如果本次调用有特殊设置 (比如临时想改 temperature)，则覆盖默认设置
        if kwargs:
            payload.update(kwargs)

        # 4. 发送请求
        try:
            # 这里的 timeout 设置为 60 秒，防止网络卡死
            response = requests.post(
                url, headers=self.headers, json=payload, timeout=600
            )

            # 设置文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.default_model
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            file_path_request = f"{output_dir}/{timestamp}_req.json"
            file_path_response = f"{output_dir}/{timestamp}_resp.json"

            try:
                with open(file_path_request, "wt", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
            except Exception as e:
                print(e)
                pass

            # 解析结果
            result = response.json()

            try:
                with open(file_path_response, "wt", encoding="utf-8") as f:
                    json.dump(
                        result,
                        f,
                        ensure_ascii=False,
                    )
            except Exception as e:
                print(e)
                pass
            
            # 检查 HTTP 错误
            response.raise_for_status()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error: 响应数据为空或格式异常: {result}"

        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            # 尝试读取 API 返回的详细错误 JSON
            if e.response is not None:
                try:
                    err_json = e.response.json()
                    if "error" in err_json:
                        error_msg = f"API Error: {err_json['error']['message']}"
                except:
                    pass
            return f"请求失败: {error_msg}"


# --- 使用示例 ---

if __name__ == "__main__":

    # 场景：初始化一个“严谨且回复简短”的 AI 客户端
    # 我们在这里直接把 temperature 和 max_tokens 设置好
    ai = SimpleAI(
        api_key="sk-mbehzontcpvsficqezgplseeyrnxqnyblhlqbtsqqkuzcewy",
        base_url="https://api.siliconflow.cn/v1/chat/completions",  # 以 DeepSeek 为例
        default_model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        # --- 在这里设置全局参数 ---
        temperature=0.1,  # 设置得很低，让它回答严谨
        max_tokens=2048,  # 限制回复长度
        frequency_penalty=0.5,  # 也可以传一些不常用的参数
    )

    print(">>> 测试 1: 使用类的默认设置 (严谨模式)")
    # 调用时非常干净，不需要传参数
    res1 = ai.chat("1+1等于几？请详细论证。")
    print(res1)

    print("\n>>> 测试 2: 临时覆盖设置 (发散模式)")
    # 比如这次我想让它写诗，需要高 temperature，可以直接覆盖类的设置
    res2 = ai.chat(
        "请不要思考,直接输出1+3+5+7+...+101的结果",
        temperature=1.2,  # 这次调用会覆盖类初始化时的 0.1
        max_tokens=500,  # 覆盖类初始化时的 100
    )
    print(res2)
