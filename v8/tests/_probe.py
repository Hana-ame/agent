"""探测 pi 是否可用（只做 command -v，不调模型）。"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess


def pi_probe() -> tuple[bool, str]:
    local = shutil.which("pi")
    if not local:
        base = pathlib.Path.home() / ".nvm" / "versions" / "node"
        if base.is_dir():
            hits = sorted(base.glob("*/bin/pi"), reverse=True)
            if hits:
                local = str(hits[0])
    if local:
        return True, f"local:{local}"

    host = os.environ.get("DAG8_SSH_HOST", "lumin@172.29.89.192")
    key = os.path.expanduser(os.environ.get("DAG8_SSH_KEY", "~/.ssh/id_rsa"))
    remote = "export PATH=$HOME/.nvm/versions/node/v22.23.1/bin:$PATH; command -v pi"
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-i", key, host, remote],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0 and "pi" in (r.stdout or ""):
            return True, f"ssh:{host}"
        return False, f"ssh:{host} 上没有 pi（{r.returncode}: {(r.stderr or '').strip()[:80]}）"
    except Exception as exc:  # noqa: BLE001
        return False, f"探测失败：{type(exc).__name__}: {exc}"
