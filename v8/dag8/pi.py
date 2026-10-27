"""v8 执行层：把一条 edge 的 prompt 交给 `pi` 这个 agent 去跑。

## 为什么用 stdin 传 prompt

`pi -p "很长的多行 prompt"` 会在多层 shell 里被转义搞坏（引号、换行、中文）。
实测 `echo "$prompt" | pi -p ...` 稳定：prompt 走管道，命令行里不留内容。

## 两种执行后端

| 模式 | 何时用 | 命令 |
|---|---|---|
| `local` | 代码就跑在 pi 所在的机器上 | `pi -p ...` |
| `ssh`   | 代码在 Windows、pi 在 WSL | `ssh host 'cd ~ && pi -p ...'` |

由 `PiConfig.ssh_host` 决定：给了就走 ssh。

## 已知故障与对策

网关有 RPM/TPM 限制，连续调用会撞 `429 inference exceeds tpm/rpm limit`。
这里做成可重试（2s/4s/8s 退避），因为这类限流是瞬时的。
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time

from dag8.backoff import jittered
from pathlib import Path
from typing import Any, List, Optional, Sequence

from dag8.config import PiConfig

__all__ = ["PiAgent", "PiError"]

_RETRYABLE = ("429", "rate_limit", "insufficient_quota", "timeout", "Temporary failure")


def _remote_path_prefix() -> str:
    """远端 PATH 前缀：动态找最新的 nvm node bin，不写死版本。"""
    return ('export PATH="$(ls -d "$HOME"/.nvm/versions/node/*/bin 2>/dev/null '
            '| sort -V | tail -n 1):$PATH"; ')


def find_pi() -> Optional[str]:
    """找 pi 可执行文件。

    先看 PATH；找不到再翻 nvm 目录 —— 非交互 shell（ssh 执行、cron）不加载 .bashrc，
    `which pi` 会落空，但 pi 其实装在 nvm 下面。
    """
    exe = shutil.which("pi")
    if exe:
        return exe
    base = Path.home() / ".nvm" / "versions" / "node"
    if base.is_dir():
        for cand in sorted(base.glob("*/bin/pi"), reverse=True):
            if cand.exists():
                return str(cand)
    return None


class PiError(RuntimeError):
    """pi 调用失败。`retryable` 决定上层要不要再试一次。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class PiLLMClient:
    """把 `PiAgent` 适配成 v7 的 LLMClientProtocol。

    v8 内的 graph 节点默认用它：v8 只有 pi 一个 LLM 后端；
    v7 独立运行时仍可注入自己的 client。
    """

    def __init__(self, *, provider: str = "laptop",
                 pi: Optional[PiConfig] = None,
                 agent: Optional[PiAgent] = None,
                 timeout: float = 180.0) -> None:
        self.provider = provider
        self.pi = pi or PiConfig()
        self.agent = agent
        self.timeout = float(timeout)

    def complete(self, prompt: str, *, model_config: Any = None,
                 system_prompt: str = "", phase: str = "edge",
                 is_clean_context: bool = True,
                 selected_inputs: Optional[Sequence[str]] = None) -> str:
        provider = str(getattr(model_config, "key", "") or self.provider)
        model = str(getattr(model_config, "model", "") or "deepseek-flash")
        text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        runner = self.agent or PiAgent(self.pi)
        return runner.run(text, provider=provider, model=model,
                          tools=False, timeout=self.timeout)

    def embed(self, text: str, *, model_config: Any = None) -> List[float]:
        try:
            from v7.llm import deterministic_semantic_embedding
            return deterministic_semantic_embedding(text)
        except ModuleNotFoundError:
            # 纯 v8 环境（未安装 v7）也能做确定性的兜底向量
            import hashlib
            import math
            cleaned = str(text).lower().strip()
            vec = [0.0] * 32
            if cleaned:
                for i in range(32):
                    h = hashlib.sha256(f"{cleaned}:{i}".encode()).digest()
                    vec[i] = (h[0] / 127.5) - 1.0
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                vec = [x / norm for x in vec]
            return vec


class PiAgent:
    def __init__(self, cfg: Optional[PiConfig] = None, **overrides) -> None:
        self.cfg = cfg or PiConfig()
        for k, v in overrides.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

    # -- 命令拼装 -----------------------------------------------------------
    def _pi_args(self, provider: str, model: str, tools: bool, thinking: Optional[str] = None) -> List[str]:
        args = [
            "pi",
            "-p",
            "--provider",
            provider,
            "--model",
            model,
            "--no-session",       # 每次调用独立，不污染会话历史
            "--mode",
            "text",
        ]
        if thinking:
            args += ["--thinking", str(thinking)]
        if not tools:
            args.append("--no-tools")   # edge 是纯文本变换，不该读写磁盘
        return args

    def _command(self, provider: str, model: str, tools: bool, thinking: Optional[str] = None) -> List[str]:
        args = self._pi_args(provider, model, tools, thinking)
        if not self.cfg.ssh_host:
            exe = find_pi()
            if exe is None:
                raise PiError(
                    "本机找不到 pi（PATH 与 ~/.nvm/versions/node/*/bin 都没有）；"
                    "若 pi 装在 WSL，请配置 pi.ssh_host"
                )
            args[0] = exe
            return args

        remote = _remote_path_prefix() + " ".join(shlex.quote(a) for a in args)
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
        if self.cfg.ssh_key:
            # subprocess 不过 shell，`~` 不会被展开，必须自己展开
            cmd += ["-i", os.path.expanduser(self.cfg.ssh_key)]
        cmd += [self.cfg.ssh_host, remote]
        return cmd

    # -- 调用 ---------------------------------------------------------------
    def run(self, prompt: str, *, provider: Optional[str] = None, model: Optional[str] = None,
            tools: Optional[bool] = None, timeout: Optional[float] = None) -> str:
        provider = provider or self.cfg.provider
        model = model or self.cfg.model
        tools = self.cfg.tools if tools is None else tools
        cmd = self._command(provider, model, tools, self.cfg.thinking)

        # `timeout` 是**总预算**，不是单次尝试的上限：
        # 旧实现最坏要等 timeout × retries（实测一次链式卡顿 637s）。
        total = float(timeout or self.cfg.timeout)
        deadline = time.monotonic() + total
        attempts = max(1, int(self.cfg.retries))
        for attempt in range(attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PiError(
                    f"pi 超时（总预算 {total:.0f}s，已尝试 {attempt} 次）",
                    retryable=False,
                )
            try:
                return self._once(cmd, prompt, min(remaining, total))
            except PiError as exc:
                if not exc.retryable or attempt == attempts - 1:
                    raise
                remaining_after = max(0.0, deadline - time.monotonic())
                backoff = min(jittered(2.0, attempt), remaining_after)
                if backoff <= 0:
                    raise PiError(
                        f"pi 超时（总预算 {total:.0f}s，重试退避已耗尽）",
                        retryable=False,
                    ) from exc
                time.sleep(backoff)
        raise PiError(f"pi 超时（总预算 {total:.0f}s）")  # pragma: no cover

    @staticmethod
    def _once(cmd: List[str], prompt: str, timeout: float) -> str:
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise PiError(f"pi 超时（{timeout}s）", retryable=True) from exc
        except FileNotFoundError as exc:
            raise PiError(f"命令不存在：{cmd[0]}") from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            retryable = any(tag in detail for tag in _RETRYABLE)
            raise PiError(f"pi 退出码 {proc.returncode}：{detail[:500]}", retryable=retryable)

        out = (proc.stdout or "").strip()
        if not out:
            # 退出码 0 但没输出：多半被上游静默掐了，值得重试一次
            raise PiError("pi 返回空内容", retryable=True)
        return out
