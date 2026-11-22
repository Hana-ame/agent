"""Tool 9: Static anti-pitfall linter for SV testbenches and verification scripts.

Consolidates three battle-tested linters from agent/verilog:
  * Clocking race checks (C1, C2): blocking assignments on interface signals at posedge
  * AXI handshake deadlock checks (R1-R4): checking VALID without READY or dropping READY early
  * Silent-pass checks (S1-S3): constant-return stubs, SKIP returning 0, count-only PASS checks
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fpga_toolbox"

from .core.types import err_result, ok_result

_IFACE_RE = re.compile(r"\b(axi|apb)\.[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)")
_WAIT_RE = re.compile(r"@\s*\(\s*posedge|@\s*\(\s*negedge|repeat\s*\(|#\s*\d")
_READY_OF = {
    "AWVALID": "AWREADY",
    "WVALID": "WREADY",
    "ARVALID": "ARREADY",
    "BVALID": "BREADY",
    "RVALID": "RREADY",
}


def scan_clocking(path: Path) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, raw in enumerate(lines):
        code = raw.split("//")[0]
        if not code.strip():
            continue
        if re.search(r"@\s*\(\s*posedge", code):
            if re.search(r"#\s*\d", code):
                continue
            if _IFACE_RE.search(code):
                issues.append({
                    "severity": "ERROR",
                    "code": "C1",
                    "path": f"{path}:{i + 1}",
                    "message": "Blocking interface drive on same line as @(posedge).",
                    "context": raw.strip(),
                })
                continue
            for k in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[k].split("//")[0]
                if not nxt.strip():
                    continue
                if _WAIT_RE.search(nxt) or re.search(r"\b(axi|apb)\.[A-Za-z_]+\s*<=", nxt):
                    break
                if _IFACE_RE.search(nxt):
                    issues.append({
                        "severity": "ERROR",
                        "code": "C1",
                        "path": f"{path}:{k + 1}",
                        "message": "Blocking interface drive immediately after @(posedge) without delay.",
                        "context": lines[k].strip(),
                    })
                    break
                if re.search(r"[;$]|begin|if\s*\(", nxt):
                    break
        if re.search(r"always\s*@\s*\(\s*posedge", code):
            depth = 0
            for k in range(i, min(i + 60, len(lines))):
                nxt = lines[k].split("//")[0]
                if k > i and re.search(r"always\s*@|endmodule", nxt):
                    break
                if re.search(r"\bbegin\b", nxt):
                    depth += 1
                if re.search(r"\bend\b", nxt) and not re.search(r"\bend\w+", nxt):
                    depth -= 1
                    if depth <= 0 and k > i:
                        break
                if k > i and _IFACE_RE.search(nxt) and "<=" not in nxt:
                    issues.append({
                        "severity": "ERROR",
                        "code": "C2",
                        "path": f"{path}:{k + 1}",
                        "message": "Blocking interface drive inside always @(posedge) block.",
                        "context": lines[k].strip(),
                    })
    return issues


def scan_handshake(path: Path) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    def _ready_raised_before(idx: int, ready_sig: str, window: int = 40) -> bool:
        for j in range(idx - 1, max(0, idx - window) - 1, -1):
            ln = lines[j].split("//")[0]
            if re.search(rf"\b{ready_sig}\b\s*(<=|=)\s*1'b0", ln):
                return False
            if re.search(rf"\b{ready_sig}\b\s*(<=|=)\s*1", ln):
                return True
        return False

    for i, raw in enumerate(lines):
        ln = raw.split("//")[0]
        if not ln.strip():
            continue
        # 检测「等 VALID 但未同时检查 READY」的握手风险。两种等价写法都要覆盖：
        #   while (!BVALID) @(posedge clk);   ← 旧实现只抓这种
        #   wait (BVALID);                    ← 第三方 TB 广泛使用，旧实现漏报
        # 两者语义相同：等待 VALID 期间未同时检查对应 READY。
        # 兼容裸信号名（bvalid）与带前缀（axi.BVALID），大小写不敏感。
        mwait = re.search(r"\bwait\s*\(([^)]*)\)", ln, re.IGNORECASE)
        if mwait:
            cond = mwait.group(1)
            has_v = re.search(r"([A-Za-z_][A-Za-z0-9_.]*VALID)", cond, re.IGNORECASE)
            has_r = re.search(r"[A-Za-z_][A-Za-z0-9_.]*READY", cond, re.IGNORECASE)
            if has_v and not has_r:
                sig = has_v.group(1)
                # 去掉可选前缀后查 READY 映射（_READY_OF 键为全大写）
                ready = _READY_OF.get(sig.split(".")[-1].upper())
                safe = bool(ready and _ready_raised_before(i, ready))
                code = "R1" if safe else "R3"
                sev = "WARNING" if safe else "ERROR"
                issues.append({
                    "severity": sev,
                    "code": code,
                    "path": f"{path}:{i + 1}",
                    "message": f"wait() waits only on {sig} without checking {ready or 'READY'}.",
                    "context": raw.strip(),
                })
        mwhile = re.search(r"while\s*\((.*)\)", ln)
        if mwhile:
            cond = mwhile.group(1)
            has_v = re.search(r"[A-Z_]+VALID", cond)
            has_r = re.search(r"[A-Z_]+READY", cond)
            if has_v and not has_r:
                sig = has_v.group(0)
                ready = _READY_OF.get(sig.replace("axi.", ""))
                safe = bool(ready and _ready_raised_before(i, ready))
                code = "R1" if safe else "R3"
                sev = "WARNING" if safe else "ERROR"
                issues.append({
                    "severity": sev,
                    "code": code,
                    "path": f"{path}:{i + 1}",
                    "message": f"while loop waits only on {sig} without checking READY.",
                    "context": raw.strip(),
                })
        m4 = re.search(r"while\s*\(\s*!\s*(?:\()?\s*axi\.([A-Z_]+VALID)", ln)
        if m4:
            sig4 = m4.group(1)
            ready4 = _READY_OF.get(sig4)
            if ready4:
                k = i
                if "begin" in ln:
                    depth = 1
                    for k in range(i + 1, min(i + 30, len(lines))):
                        body = lines[k].split("//")[0]
                        if re.search(r"\bbegin\b", body):
                            depth += 1
                        if re.search(r"\bend\b", body):
                            depth -= 1
                            if depth <= 0:
                                break
                for j in range(k + 1, min(k + 4, len(lines))):
                    nxt = lines[j].split("//")[0]
                    if re.search(r"@\s*\(|#\s*\d", nxt):
                        break
                    if re.search(rf"\b{ready4}\b\s*(<=|=)\s*1'b0", nxt):
                        issues.append({
                            "severity": "ERROR",
                            "code": "R4",
                            "path": f"{path}:{i + 1}",
                            "message": f"Drops {ready4} immediately after waiting on {sig4}.",
                            "context": raw.strip(),
                        })
                        break
        mif = re.search(r"if\s*\(\s*axi\.([A-Z_]+VALID)\s*\)", ln)
        if mif:
            sig = mif.group(1)
            ready = _READY_OF.get(sig)
            if ready:
                danger = False
                for k in range(i, min(i + 6, len(lines))):
                    nxt = lines[k].split("//")[0]
                    if k > i and re.search(r"@\s*\(|repeat\s*\(|#\s*\d", nxt):
                        break
                    if re.search(rf"\b{ready}\b\s*(<=|=)\s*1'b0", nxt):
                        danger = True
                        break
                if danger:
                    issues.append({
                        "severity": "ERROR",
                        "code": "R2",
                        "path": f"{path}:{i + 1}",
                        "message": f"Drops {ready} immediately inside if({sig}) branch.",
                        "context": raw.strip(),
                    })
                elif not _ready_raised_before(i, ready):
                    issues.append({
                        "severity": "WARNING",
                        "code": "R1",
                        "path": f"{path}:{i + 1}",
                        "message": f"Checks {sig} without visible prior {ready}=1.",
                        "context": raw.strip(),
                    })
    return issues


def scan_silent_pass(path: Path) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # S2: SKIP followed closely by return 0 / exit 0
        if re.search(r"\bSKIP\b", raw) and not stripped.startswith(("echo", "print")):
            for k in range(i, min(i + 4, len(lines))):
                if re.search(r"\b(return|exit)\s+0\b", lines[k]):
                    issues.append({
                        "severity": "WARNING",
                        "code": "S2",
                        "path": f"{path}:{k + 1}",
                        "message": "SKIP branch exits with status 0 (silent pass risk).",
                        "context": lines[k].strip(),
                    })
                    break
        # S3: Count-only PASS threshold (-ge N alongside PASS grep)
        if re.search(r"grep\s+-c.*PASS", raw) or re.search(r"passes.*-ge\s+\$?", raw):
            issues.append({
                "severity": "WARNING",
                "code": "S3",
                "path": f"{path}:{i + 1}",
                "message": "Count-only PASS check without verifying individual suite identities.",
                "context": stripped,
            })
        # S1: Shell/SV stub function immediately returning constant 0/1
        if re.match(r"^(?:function\s+)?[A-Za-z_]\w*\s*\(\)\s*\{\s*return\s+[01]\s*;?\s*\}$", stripped):
            issues.append({
                "severity": "ERROR",
                "code": "S1",
                "path": f"{path}:{i + 1}",
                "message": "Verification helper stub returns a constant value.",
                "context": stripped,
            })
    return issues


def _collect_targets(target_path: Path) -> Tuple[List[Path], List[Path]]:
    sv_files: List[Path] = []
    script_files: List[Path] = []
    if target_path.is_file():
        if target_path.suffix.lower() in (".sv", ".v"):
            sv_files.append(target_path)
        elif target_path.suffix.lower() in (".sh", ".py"):
            script_files.append(target_path)
        return sv_files, script_files
    for f in sorted(target_path.rglob("*")):
        if not f.is_file() or "__pycache__" in f.parts or ".git" in f.parts:
            continue
        suf = f.suffix.lower()
        if suf in (".sv", ".v") and (f.name.startswith("tb_") or "vip" in f.parts or "driver" in f.name):
            sv_files.append(f)
        elif suf in (".sh", ".py") and not f.name.startswith("lint_"):
            script_files.append(f)
    return sv_files, script_files


def tool_lint_tb(
    target_path: str,
    rules: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    p = Path(target_path)
    if not p.exists():
        return err_result("lint_tb", f"Target path not found: {target_path}")
    enabled = set(rules) if rules else {"clocking", "handshake", "silent_pass"}
    sv_files, script_files = _collect_targets(p)
    findings: List[Dict[str, Any]] = []
    for f in sv_files:
        if "clocking" in enabled:
            findings.extend(scan_clocking(f))
        if "handshake" in enabled:
            findings.extend(scan_handshake(f))
    for f in script_files:
        if "silent_pass" in enabled:
            findings.extend(scan_silent_pass(f))

    errors = [x for x in findings if x["severity"] == "ERROR"]
    warnings = [x for x in findings if x["severity"] == "WARNING"]
    clean = len(errors) == 0 and (not strict or len(warnings) == 0)
    data = {
        "target_path": str(p.resolve()),
        "scanned_sv_files": len(sv_files),
        "scanned_script_files": len(script_files),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "clean": clean,
        "findings": findings,
    }
    if not clean:
        # 违规时仍带 data：调用方需要结构化的 error_count/warning_count/findings
        # 汇总（与 run_mutation 等 ok=False 也带 data 的工具保持一致），
        # 否则只能从 diagnostics 重新解析，且拿不到计数字段。
        return err_result(
            "lint_tb",
            f"Lint failed with {len(errors)} error(s) and {len(warnings)} warning(s).",
            data=data,
            diagnostics=findings,
        )
    return ok_result("lint_tb", data, diagnostics=warnings)


if __name__ == "__main__":
    import sys

    from .cli import run_single_tool

    sys.exit(run_single_tool("lint_tb"))
