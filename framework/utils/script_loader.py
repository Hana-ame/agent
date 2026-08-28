"""Script loader module - Dynamic loading of external Python scripts.

Vertex scripts may export:
    on_receive(data, data_id, tags, settings) -> data   (may raise to reject)
    on_ready(all_data, settings) -> {(data_id, (tags,)): value}

Edge scripts may export:
    pre_process(data, settings) -> data
    post_process(data, settings) -> data
"""

import importlib.util
import logging
import os
from typing import Optional

logger = logging.getLogger("vertex_edge_agent.script_loader")


def load_script(script_path: str, script_name: Optional[str] = None):
    """Load a Python script as a module.

    Args:
        script_path:  Absolute or relative path to the ``.py`` file.
        script_name:  Module name (defaults to the filename stem).

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: Script does not exist.
        ImportError:       Script cannot be loaded / executed.
    """
    script_path = os.path.abspath(script_path)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    if script_name is None:
        script_name = os.path.splitext(os.path.basename(script_path))[0]

    logger.info("[ScriptLoader] Loading '%s' from %s", script_name, script_path)

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
        base_class: The base class to look for.
        default_class: The class to return if no subclass is found. Defaults to base_class.
        
    Returns:
        The found subclass, or default_class if none found.
        
    Raises:
        RuntimeError: If script fails to load.
    """
    import inspect
    if default_class is None:
        default_class = base_class
        
    try:
        module = load_script(script_path)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, base_class) and obj not in (base_class, default_class):
                return obj
        return default_class
    except Exception as exc:
        raise RuntimeError(f"Pipeline script load failed for '{script_path}': {exc}") from exc
