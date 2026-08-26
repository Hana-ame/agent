"""Script loader module - Dynamic loading of external Python scripts.

脚本加载器模块 —— 动态加载外部 Python 脚本。

Vertex scripts may export:  顶点脚本可导出的钩子：
    on_receive(data, data_id, tags, settings) -> data   (may raise to reject)
    on_ready(all_data, settings) -> {(data_id, (tags,)): value}

Edge scripts may export:    边脚本可导出的钩子：
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

    将外部 Python 脚本加载为模块对象。

    Args:
        script_path:  Absolute or relative path to the ``.py`` file.
                      脚本的绝对路径或相对路径。
        script_name:  Module name (defaults to the filename stem).
                      模块名(默认取文件名去掉扩展名)。

    Returns:
        The loaded module object.  加载后的模块对象。

    Raises:
        FileNotFoundError: Script does not exist.  脚本文件不存在。
        ImportError:       Script cannot be loaded / executed.  脚本无法加载或执行。
    """
    script_path = os.path.abspath(script_path)

    # 脚本文件必须存在
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    # 未指定模块名时，使用文件名(不含扩展名)作为模块名
    if script_name is None:
        script_name = os.path.splitext(os.path.basename(script_path))[0]

    logger.info("[ScriptLoader] Loading '%s' from %s", script_name, script_path)

    try:
        # 通过 importlib 从文件路径创建模块并执行
        spec = importlib.util.spec_from_file_location(script_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec from {script_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 记录脚本导出的可调用对象，便于调试
        callables = [
            n for n in dir(module)
            if callable(getattr(module, n)) and not n.startswith("_")
        ]
        logger.debug("[ScriptLoader] '%s' exports: %s", script_name, callables)

        return module

    except Exception as exc:
        logger.error("[ScriptLoader] Failed to load %s: %s", script_path, exc)
        raise
