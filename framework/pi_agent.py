"""PI Agent module - Interface for AI / LLM processing.

PI Agent 模块 —— AI / LLM 处理接口。

Provides an abstract base class ``PIAgent`` and two concrete implementations:

提供抽象基类 ``PIAgent`` 和两个具体实现：
* ``MockPIAgent``      – deterministic, for testing
                         确定性的 Mock 实现，用于测试
* ``ExternalPIAgent``  – delegates to an installed ``pi_agent`` package
                         委托给已安装的 ``pi_agent`` 第三方包(预留真实接入)
"""

import abc
import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("vertex_edge_agent.pi_agent")

# 禁止使用的 opencode provider（用户明确禁用 opencode-go 的模型）
FORBIDDEN_OPENCODE_PROVIDERS = ("opencode-go",)

# opencode 真实调用时的本地代理出口，共 6 个可选(均 :7890)。
# 用法: OpenCodeAgent(proxy="3") 或 proxy="http://127.0.2.4:7890" 或 proxy=["1","2"] 自动切换。
OPENCODE_PROXIES: List[str] = [
    "http://127.0.1.4:7890",
    "http://127.0.1.6:7890",
    "http://127.0.2.4:7890",
    "http://127.0.2.6:7890",
    "http://127.0.3.4:7890",
    "http://127.0.3.6:7890",
]


class PIAgent(abc.ABC):
    """Abstract base class for PI Agent integration.

    PI Agent 集成的抽象基类。所有具体实现都需继承并实现 ``process`` 方法。
    """

    @abc.abstractmethod
    async def process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> Any:
        """Process *data* through the AI agent.

        通过 AI agent 处理 *data*。

        Args:
            data:     Input data (string, dict, or any JSON-serialisable value).
                      输入数据(字符串、字典或任意可 JSON 序列化的值)。
            prompt:   The instruction / prompt.  指令 / 提示词。
            model:    Model identifier (e.g. ``"gemini-pro"``).  模型标识。
            settings: Extra settings forwarded to the agent.  透传给 agent 的额外配置。

        Returns:
            Processed result (string or JSON-serialisable value).  处理结果。
        """


class MockPIAgent(PIAgent):
    """Deterministic mock agent for testing.

    用于测试的确定性 Mock agent。

    By default it echoes data back unchanged (no model name in the result).
    Supply a custom *response_fn(data, prompt, model, settings) -> result* to override.

    默认会把数据原样回显(结果中不包含模型名)；可传入自定义的
    *response_fn(data, prompt, model, settings) -> result* 进行覆盖。
    """

    def __init__(self, response_fn: Optional[Callable] = None):
        # 可选的响应函数，用于定制 Mock 行为
        self._response_fn = response_fn

    async def process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> Any:
        logger.debug("[MockPIAgent] model=%s", model)
        logger.debug("[MockPIAgent] data=%s", repr(data)[:200])
        logger.debug("[MockPIAgent] prompt=%s", prompt[:200] if prompt else "")

        if self._response_fn:
            # 用户提供了自定义响应函数，直接调用它
            result = self._response_fn(data, prompt, model, settings)
        else:
            # 默认行为：原样回显数据(结果中不包含模型名，与真实 agent 一致)
            result = data

        logger.debug("[MockPIAgent] result=%s", repr(result)[:200])
        return result


class ExternalPIAgent(PIAgent):
    """Delegates to an installed ``pi_agent`` Python package.

    委托给已安装的 ``pi_agent`` Python 包(真实 AI 接入点)。

    Install via ``pip install pi-agent`` (or equivalent).
    可通过 ``pip install pi-agent`` 安装。

    Note: 若未安装该包，可改用 :class:`PICLIPIAgent` —— 它通过 pi 的
    命令行直接调用真实模型，无需额外安装。
    """

    async def process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> Any:
        logger.debug("[ExternalPIAgent] model=%s", model)
        try:
            # 动态导入第三方 pi_agent 包并调用其 run 接口
            import pi_agent as pa  # type: ignore[import-untyped]

            result = await pa.run(
                data=data, prompt=prompt, model=model, **(settings or {})
            )
            return result
        except ImportError:
            # 包未安装时给出明确提示
            logger.error(
                "[ExternalPIAgent] 'pi_agent' package not installed. "
                "Use MockPIAgent for testing or install the package."
            )
            raise


class PICLIPIAgent(PIAgent):
    """Real PI Agent via the ``pi`` command-line interface.

    通过 pi 命令行(真实 pi coding agent)调用真实 LLM 的 Agent。

    在非交互的 print 模式下调用 ``pi --print --no-tools --no-session``,
    把每条边的 data + prompt 发送给真实模型并返回生成的文本。

    Args:
        provider:   pi provider 名称(默认使用 pi 配置的默认 provider)。
        model:      模型名称(默认使用 pi 配置的默认模型)。
        cli:        ``pi`` 可执行文件路径。
        timeout:    单次调用的超时秒数。
        thinking:   pi 的思考级别(off/minimal/low/medium/high/...)。
        no_tools:   禁用工具，仅做纯文本生成(避免 agent 去改文件)。
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        cli: str = "pi",
        timeout: float = 180.0,
        thinking: str = "low",
        no_tools: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.cli = cli
        self.timeout = timeout
        self.thinking = thinking
        self.no_tools = no_tools

    @staticmethod
    def _fmt(data: Any) -> str:
        """把输入数据格式化为文本。"""
        if isinstance(data, str):
            return data
        return json.dumps(data, ensure_ascii=False, default=str)

    async def process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> Any:
        settings = settings or {}
        # provider 优先级: settings > 构造器
        provider = settings.get("provider") or self.provider
        # 模型优先级: 构造器显式指定 > settings > 边级 model(config)
        model = self.model or settings.get("model") or model

        # 组装要发送给 LLM 的完整文本: 提示词 + 输入数据
        text = f"{prompt}\n\n--- INPUT ---\n{self._fmt(data)}"

        # 构建 pi 命令行参数
        cmd = [self.cli, "--print", "--no-session"]
        if self.no_tools:
            cmd.append("--no-tools")
        if self.thinking:
            cmd += ["--thinking", self.thinking]
        if provider:
            cmd += ["--provider", provider]
        if model:
            cmd += ["--model", model]
        cmd.append(text)

        tag = f"[{(settings or {}).get('edge_id')}] " if (settings or {}).get("edge_id") else ""
        logger.debug("[PICLIPIAgent]%sCMD: %s", tag, " ".join(cmd))
        logger.debug(
            "[PICLIPIAgent]%scalling pi | provider=%s model=%s cli=%s",
            tag, provider, model, self.cli,
        )

        # 用子进程调用 pi,并应用超时
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError(
                f"[PICLIPIAgent] pi call timed out after {self.timeout}s "
                f"(model={model})"
            ) from None

        if proc.returncode != 0:
            raise RuntimeError(
                f"[PICLIPIAgent] pi failed (rc={proc.returncode}): "
                f"{err.decode(errors='replace')[:500]}"
            )

        result = out.decode(errors="replace").strip()
        logger.debug("[PICLIPIAgent] got %d chars from model=%s", len(result), model)
        return result


class OpenCodeAgent(PIAgent):
    """Real agent via the ``opencode`` command-line interface.

    通过 opencode 命令行(opencode coding agent)调用真实 LLM 的 Agent。
    与 :class:`PICLIPIAgent`(pi CLI) 并列的另一种后端实现。

    在非交互模式下调用 ``opencode run -m <model> <text>``，
    把每条边的 data + prompt 发送给真实模型并返回生成的文本。

    Args:
        model:      模型名，格式 ``provider/model``；默认 ``opencode-zen/hy3-free``
                    (自动归一化为 opencode CLI 可用的 ``opencode/hy3-free``)。
        cli:        ``opencode`` 可执行文件路径。
        timeout:    单次调用的超时秒数。
        variant:    opencode 模型变体(如 high/max，对应推理强度)。
        pure:       不加载外部插件，仅做纯文本生成。
        proxy:      代理配置：None=不用代理；"1".."6"=OPENCODE_PROXIES 的索引；
                    URL=单个代理；列表/逗号串=多个候选，失败自动切换。
    """

    def __init__(
        self,
        model: str = "opencode-zen/hy3-free",
        cli: str = "opencode",
        timeout: float = 240.0,
        variant: Optional[str] = None,
        pure: bool = True,
        proxy=None,
    ):
        self.model = self._normalize_model(model)  # 归一化 + 禁止校验
        self.cli = cli
        self.timeout = timeout
        self.variant = variant
        self.pure = pure
        self.proxies = self._resolve_proxies(proxy)

    # ------------------------------------------------------------------
    # Proxy helpers
    # ------------------------------------------------------------------
    @classmethod
    def _norm_proxy(cls, p: str) -> str:
        """把代理配置归一化为完整 URL。"""
        p = p.strip()
        if not p:
            return ""
        if p.isdigit() and 1 <= int(p) <= len(OPENCODE_PROXIES):
            return OPENCODE_PROXIES[int(p) - 1]
        if not p.startswith("http"):
            p = "http://" + p
        return p

    @classmethod
    def _resolve_proxies(cls, proxy) -> List[str]:
        """把 proxy 参数解析为代理 URL 列表。"""
        if proxy is None:
            return []
        if isinstance(proxy, (list, tuple)):
            return [cls._norm_proxy(p) for p in proxy if cls._norm_proxy(p)]
        if isinstance(proxy, str):
            proxy = proxy.strip()
            if proxy.lower() in ("", "none", "off", "false"):
                return []
            parts = [p for p in proxy.split(",") if p.strip()]
            return [cls._norm_proxy(p) for p in parts if cls._norm_proxy(p)]
        return []

    @staticmethod
    def _normalize_model(model: Optional[str]) -> Optional[str]:
        """把模型名归一化为 opencode CLI 能识别的名称。

        ``opencode-zen/hy3-free`` 是 pi 配置里的 provider/model 写法；
        opencode CLI 的 provider 名是 ``opencode``，因此自动映射：
        ``opencode-zen/*`` -> ``opencode/*``。
        """
        if model and model.startswith("opencode-zen/"):
            model = "opencode/" + model.split("/", 1)[1]
        
        return model

    @staticmethod
    def _fmt(data: Any) -> str:
        """把输入数据格式化为文本。"""
        if isinstance(data, str):
            return data
        return json.dumps(data, ensure_ascii=False, default=str)

    async def process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> Any:
        settings = settings or {}
        # 模型优先级: 构造器显式指定 > settings > 边级 model(config)
        model = self.model or settings.get("model") or model
        model = self._normalize_model(model)  # opencode-zen/* -> opencode/*

        # 组装要发送给 LLM 的完整文本
        text = f"{prompt}\n\n--- INPUT ---\n{self._fmt(data)}"

        # 构建 opencode 命令行参数
        cmd = [self.cli, "run"]
        if self.pure:
            cmd.append("--pure")
        if model:
            cmd += ["-m", model]
        if self.variant:
            cmd += ["--variant", self.variant]
        cmd.append(text)

        tag = f"[{(settings or {}).get('edge_id')}] " if (settings or {}).get("edge_id") else ""
        logger.debug("[OpenCodeAgent]%sCMD: %s", tag, " ".join(cmd))
        logger.debug(
            "[OpenCodeAgent]%scalling opencode | model=%s cli=%s proxies=%s",
            tag, model, self.cli, self.proxies,
        )

        # 逐个代理尝试：每个代理设置 HTTPS_PROXY 等环境变量后调用 opencode
        last_err: Optional[str] = None
        for proxy in self.proxies or [None]:
            env = os.environ.copy()
            if proxy:
                for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                            "https_proxy", "http_proxy", "all_proxy"):
                    env[key] = proxy
                logger.debug("[OpenCodeAgent] using proxy %s", proxy)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                raise TimeoutError(
                    f"[OpenCodeAgent] opencode call timed out after {self.timeout}s "
                    f"(model={model}, proxy={proxy})"
                ) from None

            if proc.returncode == 0:
                result = out.decode(errors="replace").strip()
                logger.debug(
                    "[OpenCodeAgent] got %d chars from model=%s (proxy=%s)",
                    len(result), model, proxy,
                )
                return result

            # 该代理失败，记录后尝试下一个
            err_text = err.decode(errors="replace")[:300]
            last_err = f"proxy={proxy} rc={proc.returncode}: {err_text}"
            logger.debug("[OpenCodeAgent] attempt failed: %s", last_err)

        # 所有代理都失败
        raise RuntimeError(
            f"[OpenCodeAgent] opencode failed after {len(self.proxies) or 1} "
            f"attempt(s): {last_err}"
        )
        logger.debug("[OpenCodeAgent] got %d chars from model=%s", len(result), model)
        return result
