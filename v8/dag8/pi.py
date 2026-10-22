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
from typing import List, Optional

from dag8.config import PiConfig

__all__ = ["PiAgent", "PiError"]

_RETRYABLE = ("429", "rate_limit", "insufficient_quota", "timeout", "Temporary failure")
_NODE_BIN = "$HOME/.nvm/versions/node/v22.23.1/bin"


class PiError(RuntimeError):
    """pi 调用失败。`retryable` 决定上层要不要再试一次。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class PiAgent:
    def __init__(self, cfg: Optional[PiConfig] = None, **overrides) -> None:
        self.cfg = cfg or PiConfig()
        for k, v in overrides.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

    # -- 命令拼装 -----------------------------------------------------------
    def _pi_args(self, provider: str, model: str, tools: bool) -> List[str]:
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
        if not tools:
            args.append("--no-tools")   # edge 是纯文本变换，不该读写磁盘
        return args

    def _command(self, provider: str, model: str, tools: bool) -> List[str]:
        args = self._pi_args(provider, model, tools)
        if not self.cfg.ssh_host:
            exe = shutil.which("pi")
            if exe is None:
                raise PiError("本机找不到 pi 可执行文件；若 pi 装在 WSL，请配置 pi.ssh_host")
            args[0] = exe
            return args

        remote = f"export PATH={_NODE_BIN}:$PATH; " + " ".join(shlex.quote(a) for a in args)
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
        cmd = self._command(provider, model, tools)

        last: Optional[PiError] = None
        attempts = max(1, int(self.cfg.retries))
        for attempt in range(attempts):
            try:
                return self._once(cmd, prompt, timeout or self.cfg.timeout)
            except PiError as exc:
                last = exc
                if not exc.retryable or attempt == attempts - 1:
                    raise
                time.sleep(2.0 * (2 ** attempt))
        raise last  # pragma: no cover - 循环内必 return/raise

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
