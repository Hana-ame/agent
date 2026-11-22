"""EDA environment probe (Verilator >= 5.040, slang, stdbuf) ported from agent/verilog/vlt_env.sh."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

MIN_VERILATOR_VERSION: Tuple[int, int] = (5, 40)


def parse_verilator_version(text: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"Verilator\s+(\d+)\.(\d+)", text)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def find_verilator() -> Optional[str]:
    candidates = []
    if os.environ.get("VERILATOR"):
        candidates.append(os.environ["VERILATOR"])
    if os.environ.get("VERILATOR_ROOT"):
        candidates.append(str(Path(os.environ["VERILATOR_ROOT"]) / "bin" / "verilator"))
    home = Path.home()
    candidates.append(str(home / ".local" / "verilator" / "bin" / "verilator"))
    candidates.append("/tmp/verilator/bin/verilator")
    which_v = shutil.which("verilator")
    if which_v:
        candidates.append(which_v)
    for c in candidates:
        p = Path(c)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def probe_eda_env() -> Dict[str, Any]:
    """Detect Verilator (>= 5.040), slang, and stdbuf without failing silently."""
    vbin = find_verilator()
    v_ver: Optional[Tuple[int, int]] = None
    v_ok = False
    v_msg = "verilator not found"
    if vbin:
        try:
            proc = subprocess.run([vbin, "--version"], capture_output=True, text=True, timeout=5, check=False)
            v_ver = parse_verilator_version(proc.stdout + proc.stderr)
            if v_ver is None:
                v_msg = f"could not parse verilator version from '{vbin}'"
            elif v_ver < MIN_VERILATOR_VERSION:
                v_msg = (
                    f"Verilator {v_ver[0]}.{v_ver[1]:03d} < {MIN_VERILATOR_VERSION[0]}.{MIN_VERILATOR_VERSION[1]:03d} "
                    "(older versions suffer from cross-file virtual interface silent hangs)"
                )
            else:
                v_ok = True
                v_msg = f"Verilator {v_ver[0]}.{v_ver[1]:03d}"
        except Exception as exc:
            v_msg = f"verilator probe failed: {exc}"

    slang_bin = shutil.which("slang")
    stdbuf_bin = shutil.which("stdbuf")
    return {
        "verilator": {
            "path": vbin,
            "version": f"{v_ver[0]}.{v_ver[1]:03d}" if v_ver else None,
            "meets_min_version": v_ok,
            "message": v_msg,
        },
        "slang": {"path": slang_bin, "available": bool(slang_bin)},
        "stdbuf": {"path": stdbuf_bin, "available": bool(stdbuf_bin)},
    }
