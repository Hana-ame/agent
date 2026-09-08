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

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("vertex_edge_agent.script_loader")

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)


def load_script(script_path: str, script_name: Optional[str] = None):
    """Load a Python script as a module from any working directory.

    Args:
        script_path:  Absolute or relative path to the ``.py`` file.
        script_name:  Module name (defaults to the filename stem).

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: Script does not exist.
        ImportError:       Script cannot be loaded / executed.
    """
    # Resolve path: if relative and doesn't exist in current working dir, check repo root
    if not os.path.isabs(script_path) and not os.path.exists(script_path):
        repo_candidate = os.path.join(_REPO_ROOT, script_path)
        if os.path.exists(repo_candidate):
            script_path = repo_candidate

    script_path = os.path.abspath(script_path)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    # Ensure script directory and repository root are in sys.path
    script_dir = os.path.dirname(script_path)
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    if script_name is None:
        script_name = os.path.splitext(os.path.basename(script_path))[0]

    logger.debug("[ScriptLoader] Loading '%s' from %s", script_name, script_path)

    try:
        spec = importlib.util.spec_from_file_location(script_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec from {script_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Log exported callables
        callables = [
            n for n in dir(module)
            if callable(getattr(module, n)) and not n.startswith("_")
        ]
        logger.debug("[ScriptLoader] '%s' exports: %s", script_name, callables)

        return module

    except Exception as exc:
        logger.error("[ScriptLoader] Failed to load %s: %s", script_path, exc)
        raise

def load_class_from_script(script_path: str, base_class: type, default_class: type = None) -> type:
    """Load a script and find a subclass of base_class.

    Args:
        script_path: Path to the python script.
        base_class: The base class the found class must subclass.
        default_class: Either a class NAME (str) to look up explicitly (e.g.
            MapEdge pipeline steps pass ``"SummarizeEdge"``), or a fallback
            class type to return when auto-discovery finds nothing.
            Defaults to base_class.

    Returns:
        The found subclass, or the fallback if none found.

    Raises:
        RuntimeError: If script fails to load.
    """
    import inspect
    if default_class is None:
        default_class = base_class

    try:
        module = load_script(script_path)

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
