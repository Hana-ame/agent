"""Archive module - Serialize a graph run to a portable JSON document.

存档(Archive)模块 —— 把一次图执行(图定义 + 初始输入 + 运行结果)序列化成
可移植的 JSON 文档，并可保存到磁盘。这是框架自身的能力，示例只需调用它，
无需自己实现落盘逻辑。

一个完整的「存档」由三部分组成：
    ① 初始输入快照 —— 运行前每个顶点的初始数据(由 snapshot_initial_data 采集)
    ② 运行结果     —— Executor 返回的 ExecutionResult
    ③ 图定义       —— Graph(顶点设置/脚本/边定义等)

提供了两个函数：
    snapshot_initial_data(graph)  运行前采集每个顶点的初始数据快照
    build_archive(...)            把上面三部分合并成一个可 JSON 化的文档
    save_archive(archive, path)   把存档文档写入磁盘(新建文件)
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .graph import Graph
from .executor import ExecutionResult

logger = logging.getLogger("vertex_edge_agent.archive")


def snapshot_initial_data(graph: Graph) -> Dict[str, Dict]:
    """Capture each vertex's initial data store before execution.

    运行前采集每个顶点的初始数据快照，供存档时作为「输入侧」记录。
    返回 {vertex_id: {(edge_id, tags): value}}。

    Args:
        graph: The graph whose vertices' initial data to snapshot.

    Returns:
        Mapping of vertex id -> raw internal data store dict.
    """
    return {vid: dict(v._data_store) for vid, v in graph.vertices.items()}


def _init_items(store: Dict) -> List[Dict]:
    """把内部数据存储键 ((edge_id, tags) -> value) 转成可读的记录条目。

    store 的键是 (edge_id, tuple(tags))，这里转换成
    [{"data_id": edge_id, "tags": [...], "value": value}, ...] 便于阅读。
    """
    return [{"data_id": k[0], "tags": list(k[1]), "value": v}
            for k, v in store.items()]


def build_archive(
    graph: Graph,
    result: ExecutionResult,
    init_snapshot: Optional[Dict[str, Dict]] = None,
    agent_desc: str = "",
) -> Dict:
    """Build a serializable archive document from a graph run.

    把「图定义 + 初始输入快照 + 运行结果」合并成一个可 JSON 化的存档文档。
    每个顶点 = 输入(initial_data) + 运行结果(state/data/error) 合并成一条；
    每条边 = 定义(read_tags/set_tags/prompt/model) + 执行结果(result)。

    Args:
        graph:          The executed Graph.  已执行的图。
        result:         ExecutionResult returned by Executor.run().
                        Executor 返回的执行结果。
        init_snapshot:  Optional per-vertex initial data snapshot
                        (from snapshot_initial_data). 缺省为空。
        agent_desc:     Human-readable agent description for metadata.
                        用于元信息的 agent 描述(如 "opencode http://...")。

    Returns:
        A JSON-serializable archive dict.  可 JSON 序列化的存档字典。
    """
    init_snapshot = init_snapshot or {}

    vertices = []
    for vid, v in graph.vertices.items():
        vres = result.vertex_results.get(vid, {})
        vertices.append({
            "id": vid,
            "settings": v.settings,
            "script": v.script_path,
            "initial_data": _init_items(init_snapshot.get(vid, {})),
            "state": vres.get("state"),
            "data": vres.get("data", {}),
            "error": vres.get("error"),
        })

    edges = []
    for e in graph.edges.values():
        edges.append({
            "id": e.id,
            "source": e.source_id,
            "destination": e.destination_id,
            "read_tags": e.read_tags,
            "set_tags": e.set_tags,
            "prompt": e.prompt,
            "model": e.model,
            "result": result.edge_results.get(e.id),
        })

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "success": result.success,
            "execution_time_s": round(result.execution_time, 3),
            "graph": f"V={len(vertices)}, E={len(edges)}",
            "agent": agent_desc,
        },
        "vertices": vertices,
        "edges": edges,
        "errors": result.errors,
    }


def save_archive(archive: Dict, out_path: str) -> str:
    """Write an archive document to disk as JSON.

    把存档文档写入磁盘(新建/覆盖目标文件)。返回写入的绝对路径。

    Args:
        archive:  Archive dict from build_archive().  存档文档。
        out_path: Destination JSON file path.  目标 JSON 文件路径。

    Returns:
        The absolute path written to.  写入的绝对路径。
    """
    out_path = os.path.abspath(out_path)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(archive, fh, ensure_ascii=False, indent=2, default=str)
    logger.info(
        "[Archive] Saved %d vertices, %d edges -> %s",
        len(archive.get("vertices", [])), len(archive.get("edges", [])), out_path,
    )
    return out_path
