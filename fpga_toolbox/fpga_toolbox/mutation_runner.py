"""Tool 11: Parameterized RTL mutation runner with try/finally restore and residual-mutation guard.

Ported from agent/verilog/axi/mutate.py:
  * Refuses to run if a previous un-restored mutation is detected (residual guard, E030)
  * Always restores original RTL in try/finally even on crashes or timeouts
  * Quantifies mutation score = killed / total
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fpga_toolbox"

from .core.types import err_result, ok_result


def load_mutations(spec_path: Path) -> List[Dict[str, str]]:
    text = spec_path.read_text(encoding="utf-8")
    raw = json.loads(text) if spec_path.suffix.lower() == ".json" else yaml.safe_load(text)
    items = raw.get("mutations", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("Mutation spec must be a list or mapping with 'mutations' list.")
    out: List[Dict[str, str]] = []
    for idx, m in enumerate(items):
        if not isinstance(m, dict):
            continue
        out.append({
            "id": str(m.get("id", f"M{idx + 1:02d}")),
            "file": str(m["file"]),
            "old": str(m["old"]),
            "new": str(m["new"]),
            "desc": str(m.get("desc", "")),
        })
    return out


def check_residual_mutations(workdir: Path, mutations: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Detect whether any target file already has `new` without `old` (unrestored mutation)."""
    residuals: List[Dict[str, Any]] = []
    for m in mutations:
        fp = workdir / m["file"]
        if not fp.is_file():
            continue
        txt = fp.read_text(encoding="utf-8", errors="replace")
        if m["old"] not in txt and m["new"] in txt:
            residuals.append({
                "severity": "ERROR",
                "code": "E030",
                "path": m["file"],
                "message": f"Residual unrestored mutation detected for {m['id']} ({m['desc']}).",
            })
    return residuals


def tool_run_mutation(
    workdir: str,
    mutation_spec: str,
    test_cmd: List[str],
    min_score: float = 80.0,
    timeout_sec: int = 120,
    only_id: Optional[str] = None,
) -> Dict[str, Any]:
    # 同 run_sim：MCP / CLI 可能把数字序列化成字符串，边界处显式转换，
    # 避免 subprocess.run(timeout=str) 与 score < min_score 抛 TypeError。
    try:
        min_score = float(min_score)
    except (TypeError, ValueError):
        return err_result("run_mutation", f"min_score must be a number, got {min_score!r}")
    try:
        timeout_sec = int(timeout_sec)
    except (TypeError, ValueError):
        return err_result("run_mutation", f"timeout_sec must be an integer, got {timeout_sec!r}")

    wd = Path(workdir)
    sp = Path(mutation_spec)
    if not wd.is_dir():
        return err_result("run_mutation", f"Working directory not found: {workdir}")
    if not sp.is_file():
        return err_result("run_mutation", f"Mutation spec not found: {mutation_spec}")
    if not test_cmd:
        return err_result("run_mutation", "test_cmd must be a non-empty command list.")

    try:
        mutations = load_mutations(sp)
    except Exception as exc:
        return err_result("run_mutation", f"Invalid mutation spec: {exc}")

    if only_id:
        mutations = [m for m in mutations if m["id"] == only_id]
        if not mutations:
            return err_result("run_mutation", f"Mutation id '{only_id}' not found.")

    residuals = check_residual_mutations(wd, mutations)
    if residuals:
        return err_result(
            "run_mutation",
            f"Refusing to run: {len(residuals)} residual unrestored mutation(s) detected (exit-4 guard).",
            diagnostics=residuals,
        )

    results: List[Dict[str, Any]] = []
    killed = 0
    for m in mutations:
        fp = wd / m["file"]
        if not fp.is_file():
            results.append({"id": m["id"], "status": "ERROR", "detail": f"Missing file {m['file']}"})
            continue
        original = fp.read_text(encoding="utf-8")
        if m["old"] not in original:
            results.append({"id": m["id"], "status": "SKIP_UNMATCHED", "detail": "Pattern 'old' not found."})
            continue
        mutated = original.replace(m["old"], m["new"], 1)
        try:
            fp.write_text(mutated, encoding="utf-8")
            proc = subprocess.run(test_cmd, cwd=str(wd), capture_output=True, text=True, timeout=timeout_sec, check=False)
            if proc.returncode != 0:
                killed += 1
                results.append({"id": m["id"], "status": "KILLED", "returncode": proc.returncode, "desc": m["desc"]})
            else:
                results.append({"id": m["id"], "status": "SURVIVED", "returncode": 0, "desc": m["desc"]})
        except subprocess.TimeoutExpired:
            killed += 1
            results.append({"id": m["id"], "status": "KILLED_TIMEOUT", "desc": m["desc"]})
        finally:
            fp.write_text(original, encoding="utf-8")

    total = len(mutations)
    score = round((killed / total) * 100.0, 2) if total else 0.0
    data = {
        "workdir": str(wd.resolve()),
        "total": total,
        "killed": killed,
        "survived": total - killed,
        "mutation_score": score,
        "min_score": min_score,
        "results": results,
    }
    if score < min_score:
        return err_result(
            "run_mutation",
            f"Mutation score {score}% is below threshold {min_score}%.",
            diagnostics=[{"severity": "ERROR", "code": "E031", "path": "mutation_score", "message": f"{killed}/{total} killed ({score}% < {min_score}%)."}],
        )
    return ok_result("run_mutation", data)


if __name__ == "__main__":
    import sys

    from .cli import run_single_tool

    sys.exit(run_single_tool("run_mutation"))
