"""Tool 3: Copy resolved RTL/header files into an isolated sandbox with SHA-256 checksums.

Safety contract:
  * The sandbox may never be the project root, nor an ancestor of it.
  * Cleaning is only performed on a directory carrying the sandbox marker,
    unless the caller explicitly passes ``force=True``.
  * Relative directory layout is preserved, so ``rtl/a/common.sv`` and
    ``rtl/b/common.sv`` stay distinct and ``include`` paths keep working.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fpga_toolbox"

from .core.types import err_result, ok_result

MARKER_NAME = ".fpga_sandbox"


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _clear_dir(target: Path) -> None:
    for entry in target.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def tool_copy_sandbox(
    project_root: str,
    relative_files: List[str],
    sandbox_dir: str,
    clean: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    dst_root = Path(sandbox_dir).resolve()

    if not root.is_dir():
        return err_result("copy_sandbox", f"Project root not found: {root}")
    if not relative_files:
        return err_result("copy_sandbox", "relative_files is empty; nothing to stage.")

    if dst_root == root:
        return err_result("copy_sandbox", f"Refusing to use the project root as sandbox: {dst_root}")
    if _is_within(root, dst_root):
        return err_result(
            "copy_sandbox",
            f"Refusing to stage into '{dst_root}': it is an ancestor of the project root '{root}'.",
        )
    if dst_root.parent == dst_root:
        return err_result("copy_sandbox", f"Refusing to use a filesystem root as sandbox: {dst_root}")
    if dst_root == Path.home().resolve():
        return err_result("copy_sandbox", f"Refusing to use the home directory as sandbox: {dst_root}")

    if dst_root.exists():
        if not dst_root.is_dir():
            return err_result("copy_sandbox", f"Sandbox path exists but is not a directory: {dst_root}")
        if clean and any(dst_root.iterdir()):
            marked = (dst_root / MARKER_NAME).is_file()
            if not marked and not force:
                return err_result(
                    "copy_sandbox",
                    f"Refusing to clean '{dst_root}': it is not empty and carries no {MARKER_NAME} marker. "
                    "Pass force=True if you really intend to wipe it.",
                )
            _clear_dir(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / MARKER_NAME).write_text(
        "Created by fpga_toolbox.copy_sandbox. Safe to clean.\n", encoding="utf-8"
    )

    copied: List[Dict[str, Any]] = []
    seen = set()
    for rel in relative_files:
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            return err_result("copy_sandbox", f"Rejected non-portable relative path: {rel}")
        src = root / rel_path
        if not src.is_file():
            return err_result("copy_sandbox", f"Source file missing: {src}")

        key = str(rel_path).replace("\\", "/")
        if key in seen:
            continue
        seen.add(key)

        dst = dst_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({
            "source_rel": key,
            "staged_rel": str(dst.relative_to(dst_root)).replace("\\", "/"),
            "staged_path": str(dst.resolve()),
            "filename": dst.name,
            "sha256_prefix": hashlib.sha256(dst.read_bytes()).hexdigest()[:16],
        })

    return ok_result("copy_sandbox", {
        "sandbox_dir": str(dst_root.resolve()),
        "source_project_untouched": True,
        "layout_preserved": True,
        "file_count": len(copied),
        "files": copied,
    })


if __name__ == "__main__":
    import sys

    from .cli import run_single_tool

    sys.exit(run_single_tool("copy_sandbox"))
