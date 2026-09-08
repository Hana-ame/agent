"""Script loader module - Dynamic loading of external Python scripts.

A script is a Python file that defines a **subclass** of the base type
(``Vertex``, ``Edge``, ``MapEdge``, …). The framework loads the module,
finds the subclass, and instantiates it. Custom behaviour lives in the
methods the subclass overrides:

    Vertex subclass may override:
        on_receive(self, data, channel, settings) -> data   (may raise to reject)
        on_ready(self, all_data, settings) -> {(data_id, (tags,)): value}

    Edge subclass may override:
        pre_process(self, data, settings) -> data
        post_process(self, result, settings) -> result

Scripts may be referenced as ``"path/to/script.py"`` (auto-discover the
unique subclass) or ``"path/to/script.py:ClassName"`` (explicit).
"""

import hashlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("vertex_edge_agent.script_loader")

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)

#: Extra directories that scripts may be loaded from (``os.pathsep``-separated).
SCRIPT_ROOTS_ENV = "VEA_SCRIPT_ROOTS"

#: Prefixes that indicate an inline Python expression rather than a script path.
_INLINE_SCRIPT_PREFIXES = ("lambda ", "lambda:", "def ", "import ", "__import__", "exec(", "eval(")


#: (absolute path, mtime) -> loaded module. Avoids re-executing a script for
#: every edge that references it (which also keeps class identity stable).
_MODULE_CACHE: Dict[Tuple[str, float], Any] = {}


class ScriptNotAllowedError(PermissionError):
    """Raised when a script reference falls outside the allowed roots or is inline code."""


def get_allowed_script_roots(extra: Optional[Iterable[Union[str, Path]]] = None) -> List[Path]:
    """Return the directories a script may be loaded from.

    Defaults to the repository root plus any directories listed in
    ``VEA_SCRIPT_ROOTS`` (``os.pathsep``-separated) and any caller-supplied
    ``extra`` roots. The current working directory is deliberately **not** a
    default root: confinement must not depend on where the process was started.

    This is the strict set used to validate **untrusted** references (see
    :func:`validate_script_reference`). :func:`load_script` only enforces the
    configured roots (``VEA_SCRIPT_ROOTS``) so that library callers can still
    load trusted scripts from their own project directories.
    """
    roots: List[Path] = []
    for raw in (extra or []):
        roots.append(Path(raw))
    roots.extend(_env_script_roots())
    roots.append(Path(_REPO_ROOT))

    resolved: List[Path] = []
    for root in roots:
        try:
            r = root.resolve()
        except OSError:  # pragma: no cover - defensive
            continue
        if r not in resolved:
            resolved.append(r)
    return resolved


def _env_script_roots() -> List[Path]:
    """Return the roots configured via ``VEA_SCRIPT_ROOTS``."""
    roots: List[Path] = []
    for part in (os.environ.get(SCRIPT_ROOTS_ENV, "") or "").split(os.pathsep):
        if part.strip():
            roots.append(Path(part.strip()))
    return roots


def is_within(path: Union[str, Path], roots: Sequence[Path]) -> bool:
    """Return True when ``path`` resolves inside one of ``roots``."""
    try:
        target = Path(path).resolve()
    except OSError:  # pragma: no cover - defensive
        return False
    for root in roots:
        if target == root or root in target.parents:
            return True
    return False


def validate_script_reference(
    script: Optional[Union[str, object]],
    allowed_roots: Optional[Sequence[Path]] = None,
) -> None:
    """Reject inline code and out-of-root script paths.

    Raises:
        ScriptNotAllowedError: The reference is inline code, absolute outside the
            allowed roots, or escapes them via ``..``.
    """
    if script is None or callable(script):
        return
    if not isinstance(script, str):
        raise ScriptNotAllowedError(f"Unsupported script reference type: {type(script).__name__}")

    stripped = script.strip()
    if not stripped:
        return

    lowered = stripped.lower()
    if any(lowered.startswith(prefix) for prefix in _INLINE_SCRIPT_PREFIXES):
        raise ScriptNotAllowedError(
            "Inline script expressions are not allowed. Provide a script path inside an allowed root."
        )

    path_part = stripped.split(":", 1)[0]
    roots = list(allowed_roots) if allowed_roots else get_allowed_script_roots()

    if os.path.isabs(path_part):
        if not is_within(path_part, roots):
            raise ScriptNotAllowedError(f"Script path is outside the allowed roots: {path_part}")
        return

    # Relative: must resolve inside one of the roots (guards against ../ escapes).
    if not any(is_within(root / path_part, roots) for root in roots):
        raise ScriptNotAllowedError(f"Script path escapes the allowed roots: {path_part}")


def load_script(
    script_path: str,
    script_name: Optional[str] = None,
    allowed_roots: Optional[Sequence[Path]] = None,
):
    """Load a Python script as a module from any working directory.

    Args:
        script_path:  Absolute or relative path to the ``.py`` file.
        script_name:  Module name (defaults to the filename stem).
        allowed_roots: Directories the script must live under. When omitted the
            configured ``VEA_SCRIPT_ROOTS`` are enforced (if any); otherwise any
            path is allowed, because this is the trusted library entry point.
            Untrusted references must be screened with
            :func:`validate_script_reference` first.

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: Script does not exist.
        ScriptNotAllowedError: Script is outside the enforced roots.
        ImportError:       Script cannot be loaded / executed.
    """
    # Resolve a relative path against the current working directory first, then the
    # repository root, then any configured script roots. Without the last step a
    # deployment that narrows scripts to ``--script-root ./my_edges`` could not
    # reference them as ``"my_edge.py:MyEdge"``.
    if not os.path.isabs(script_path) and not os.path.exists(script_path):
        search_roots: List[Union[str, Path]] = [_REPO_ROOT]
        search_roots.extend(
            allowed_roots if allowed_roots is not None else _env_script_roots()
        )
        for root in search_roots:
            candidate = os.path.join(str(root), script_path)
            if os.path.exists(candidate):
                script_path = candidate
                break

    script_path = os.path.abspath(script_path)

    if allowed_roots is not None:
        roots = list(allowed_roots)
    else:
        roots = _env_script_roots()
    if roots and not is_within(script_path, roots):
        raise ScriptNotAllowedError(f"Script path is outside the allowed roots: {script_path}")

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    if script_name is None:
        script_name = os.path.splitext(os.path.basename(script_path))[0]

    logger.debug("[ScriptLoader] Loading '%s' from %s", script_name, script_path)

    try:
        cache_key: Tuple[str, float] = (script_path, os.path.getmtime(script_path))
    except OSError:  # pragma: no cover - defensive
        cache_key = (script_path, 0.0)
    cached = _MODULE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Register under a collision-free key so dataclasses/pickle inside the script
    # work, and expose the script directory only for the duration of the exec so
    # sibling imports resolve without permanently polluting sys.path.
    module_key = f"_vea_script_{script_name}_{hashlib.sha1(script_path.encode()).hexdigest()[:8]}"
    script_dir = os.path.dirname(script_path)
    path_inserted = False
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        path_inserted = True

    try:
        spec = importlib.util.spec_from_file_location(module_key, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec from {script_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_key] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_key, None)
            raise

        # Log exported callables
        callables = [
            n for n in dir(module)
            if callable(getattr(module, n)) and not n.startswith("_")
        ]
        logger.debug("[ScriptLoader] '%s' exports: %s", script_name, callables)
        _MODULE_CACHE[cache_key] = module
        return module

    except Exception as exc:
        logger.error("[ScriptLoader] Failed to load %s: %s", script_path, exc)
        raise
    finally:
        if path_inserted:
            try:
                sys.path.remove(script_dir)
            except ValueError:  # pragma: no cover - defensive
                pass


def load_class_from_script(
    script_path: str,
    base_class: type,
    default_class: type = None,
    allowed_roots: Optional[Sequence[Path]] = None,
) -> type:
    """Load a script and find a subclass of base_class.

    Args:
        script_path: Path to the python script.
        base_class: The base class the found class must subclass.
        default_class: Either a class NAME (str) to look up explicitly (e.g.
            MapEdge pipeline steps pass ``"SummarizeEdge"``), or a fallback
            class type to return when auto-discovery finds nothing.
            Defaults to base_class.
        allowed_roots: Directories the script must live under (forwarded to
            :func:`load_script`). Omit to enforce ``VEA_SCRIPT_ROOTS``.

    Returns:
        The found subclass, or the fallback if none found.

    Raises:
        RuntimeError: If script fails to load.
    """
    import inspect
    if default_class is None:
        default_class = base_class

    try:
        module = load_script(script_path, allowed_roots=allowed_roots)

        # Explicit class name requested -> find the class with that name
        # (previously this argument was silently ignored and the first
        # subclass of base_class was returned, so "script.py:SummarizeEdge"
        # could actually load FetchEdge).
        if isinstance(default_class, str):
            requested = getattr(module, default_class, None)
            if requested is not None and inspect.isclass(requested) and issubclass(requested, base_class):
                return requested
            logger.warning(
                "[ScriptLoader] %s does not contain a %s subclass named %s, falling back to %s -- custom behavior will not execute.",
                script_path, default_class, base_class.__name__, base_class.__name__,
            )
            return base_class

        # Auto-discover: first subclass of base_class in the script.
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, base_class) and obj not in (base_class, default_class):
                return obj

        logger.warning(
            "[ScriptLoader] %s has no %s subclass, falling back to %s -- custom behavior will not execute.\n"
            "        To customize, define a %s subclass in the script.",
            script_path, base_class.__name__, default_class.__name__, base_class.__name__,
        )
        return default_class
    except Exception as exc:
        raise RuntimeError(
            f"Script load failed for '{script_path}': {exc}"
        ) from exc
