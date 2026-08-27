"""Graph module - Load and manage the computation graph from JSON.

图(Graph)模块 —— 从 JSON 加载并管理计算图。

A Graph is a DAG of Vertex nodes connected by Edge arrows.  It is loaded
from a JSON configuration and validated for referential integrity and
acyclicity before execution.

计算图是由 Edge 连接的 Vertex 节点构成的有向无环图(DAG)。
它从 JSON 配置加载，并在执行前校验引用完整性与无环性。
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .vertex import Vertex
from .edge import Edge
from .script_loader import load_script

logger = logging.getLogger("vertex_edge_agent.graph")


class Graph:
    """DAG of vertices and edges loaded from JSON configuration.

    从 JSON 配置加载的顶点与边的 DAG。

    JSON schema:   JSON 配置结构示意::

        {
          "metadata": { ... },                      // 元信息(名称、描述等)
          "vertices": [
            {
              "id": "v1",                           // 顶点 ID
              "settings": {},                       // 任意配置
              "script": "path/to/script.py",        // 可选：顶点脚本
              "initial_data": [                     // 可选：初始数据
                {"data_id": "text", "tags": ["en"], "value": "hello"}
              ]
            }
          ],
          "edges": [
            {
              "id": "e1",                           // 边 ID
              "source": "v1",                       // 源顶点
              "destination": "v2",                  // 目标顶点
              "data_id": "text",                    // 数据键
              "tags": ["en"],                       // 标签
              "prompt": "Summarise this:",          // 提示词
              "model": "gemini-pro",                // 模型
              "settings": {},                       // 可选：边配置
              "script": "path/to/edge_script.py"    // 可选：边脚本
            }
          ]
        }
    """

    def __init__(self):
        self.vertices: Dict[str, Vertex] = {}  # 顶点 ID -> Vertex
        self.edges: Dict[str, Edge] = {}       # 边 ID -> Edge
        self.metadata: Dict[str, Any] = {}     # 图元信息

    # ------------------------------------------------------------------
    # Construction  构建
    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, json_path: str) -> "Graph":
        """Load graph from a JSON file.

        从 JSON 文件加载图。
        """
        logger.debug("[Graph] Loading from %s", json_path)
        with open(json_path, "r") as fh:
            config = json.load(fh)
        # 以 JSON 文件所在目录为基准解析脚本相对路径
        base_dir = os.path.dirname(os.path.abspath(json_path))
        return cls.from_dict(config, base_dir=base_dir)

    @classmethod
    def from_dict(cls, config: Dict, base_dir: Optional[str] = None) -> "Graph":
        """Build a graph from a configuration dict.

        从配置字典构建图。
        """
        graph = cls()
        graph.metadata = config.get("metadata", {})
        base_dir = base_dir or os.getcwd()

        # --- 创建顶点 ---
        for vc in config.get("vertices", []):
            # 解析脚本路径：相对路径基于 base_dir
            script = vc.get("script")
            if script and not os.path.isabs(script):
                script = os.path.join(base_dir, script)

            vertex = Vertex(
                vertex_id=vc["id"],
                settings=vc.get("settings", {}),
                script_path=script,
                initial_data=vc.get("initial_data"),
            )

            # 若配置了脚本则加载并挂载
            if script:
                try:
                    vertex.set_script_module(load_script(script))
                except Exception as exc:
                    logger.error(
                        "[Graph] Script load failed for vertex '%s': %s", vertex.id, exc
                    )

            graph.vertices[vertex.id] = vertex

        # --- 创建边 ---
        for ec in config.get("edges", []):
            # 同样解析脚本相对路径
            script = ec.get("script")
            if script and not os.path.isabs(script):
                script = os.path.join(base_dir, script)

            edge = Edge(
                edge_id=ec["id"],
                source_id=ec["source"],
                destination_id=ec["destination"],
                prompt=ec.get("prompt", ""),
                model=ec.get("model", "default"),
                settings=ec.get("settings", {}),
                script_path=script,
                # 读/写 tag 与纯搬运：从 config 读取
                read_tags=ec.get("read_tags") or ec.get("tags", []),
                set_tags=ec.get("set_tags") or ec.get("tags", []),
                passthrough=bool(ec.get("passthrough", False)),
            )

            if script:
                try:
                    edge.set_script_module(load_script(script))
                except Exception as exc:
                    logger.error(
                        "[Graph] Script load failed for edge '%s': %s", edge.id, exc
                    )

            graph.edges[edge.id] = edge

            # 在顶点上登记边的关联关系
            if edge.source_id in graph.vertices:
                graph.vertices[edge.source_id].outgoing_edges.append(edge.id)
            else:
                logger.error(
                    "[Graph] Edge '%s' references unknown source '%s'",
                    edge.id, edge.source_id,
                )

            if edge.destination_id in graph.vertices:
                dest = graph.vertices[edge.destination_id]
                dest.incoming_edges.append(edge.id)
                # 目标顶点所需输入数量 = 入边数量
                dest.required_input_count = len(dest.incoming_edges)
            else:
                logger.error(
                    "[Graph] Edge '%s' references unknown destination '%s'",
                    edge.id, edge.destination_id,
                )

        # 构建完成后统一校验
        graph.validate()
        logger.debug(
            "[Graph] Loaded %d vertices, %d edges",
            len(graph.vertices), len(graph.edges),
        )
        return graph

    # ------------------------------------------------------------------
    # Validation  校验
    # ------------------------------------------------------------------
    def validate(self):
        """Validate referential integrity and acyclicity.

        校验引用完整性与无环性(DAG)。
        任何不满足条件的错误都会以 ValueError 抛出。
        """
        errors: List[str] = []

        # 校验每条边的源 / 目标顶点是否存在
        for edge in self.edges.values():
            if edge.source_id not in self.vertices:
                errors.append(
                    f"Edge '{edge.id}': source '{edge.source_id}' not found"
                )
            if edge.destination_id not in self.vertices:
                errors.append(
                    f"Edge '{edge.id}': destination '{edge.destination_id}' not found"
                )

        # 环检测(DFS)：用 visited 记录已访问，stack 记录当前递归栈
        visited: set = set()
        stack: set = set()

        def _dfs(vid: str) -> bool:
            """返回 True 表示发现环。"""
            visited.add(vid)
            stack.add(vid)
            for eid in self.vertices[vid].outgoing_edges:
                nxt = self.edges[eid].destination_id
                if nxt not in self.vertices:
                    continue  # 缺失顶点已在上面单独捕获
                if nxt not in visited:
                    if _dfs(nxt):
                        return True
                elif nxt in stack:
                    # 邻居在递归栈中 => 存在环
                    return True
            stack.discard(vid)
            return False

        # 从每个尚未访问的顶点出发做 DFS
        for vid in self.vertices:
            if vid not in visited:
                if _dfs(vid):
                    errors.append("Graph contains a cycle (must be a DAG)")
                    break

        # 汇总所有错误并抛出
        if errors:
            for e in errors:
                logger.error("[Graph] Validation: %s", e)
            raise ValueError(f"Graph validation failed: {'; '.join(errors)}")

        logger.debug("[Graph] Validation passed ✓")

    # ------------------------------------------------------------------
    # Queries  查询
    # ------------------------------------------------------------------
    def get_source_vertices(self) -> List[Vertex]:
        """Vertices with no incoming edges (entry points).

        返回没有入边的顶点(入口点)。
        """
        return [v for v in self.vertices.values() if v.is_source()]

    def get_sink_vertices(self) -> List[Vertex]:
        """Vertices with no outgoing edges (exit points).

        返回没有出边的顶点(出口点)。
        """
        return [v for v in self.vertices.values() if v.is_sink()]

    def get_outgoing_edges(self, vertex_id: str) -> List[Edge]:
        """返回指定顶点的所有出边。"""
        return [self.edges[eid] for eid in self.vertices[vertex_id].outgoing_edges]

    def get_incoming_edges(self, vertex_id: str) -> List[Edge]:
        """返回指定顶点的所有入边。"""
        return [self.edges[eid] for eid in self.vertices[vertex_id].incoming_edges]

    def __repr__(self):
        return f"Graph(V={len(self.vertices)}, E={len(self.edges)})"
