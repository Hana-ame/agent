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
