"""VEA V5 有向无环图容器模块 (DAG).

彻底抛弃 V4 中繁琐的 `attributes=["start", "end"]` 手动标注与复杂的 SQLite 同步样板代码。
在 VEA V5 中：
- 你可以用最符合人类直觉的 `dag.add_step("x1", "y1", "z1")` 一句一句往图里扔 `V-E-V`；
- 起点（Roots，入度为 0 的顶点）与终点（Sinks，出度为 0 的顶点）由 DAG 自动推导；
- 当不同句子中的顶点名字不完全一致但语义相近（例如 `"商品价格列表"` 与 `"商品价格清单"`）时，
  调用 `dag.auto_link_similar_vertices()` 即可自动计算相似度并用 `VirtualEdge`（虚拟边）把它们连成完整可运行的 DAG；
- 内置成环保护（若插入某条边导致环路，自动回滚并报错）与增量脏节点重置（修改任意边后，仅重置其下游顶点）。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from vea_v5.primitives import Edge, MergeMode, Triplet, Vertex, VertexStatus, VirtualEdge


def default_text_similarity(text_a: str, text_b: str) -> float:
    """计算两个顶点名称/描述之间的综合语义相似度（0.0 ~ 1.0）。

    结合了：
    1. 归一化编辑/子串包含关系（处理如 `"z1"` 与 `"z1_output"`、`"营收数据"` 与 `"营收与成本数据"`）；
    2. 字符级 1-gram + 2-gram 余弦相似度（对中文短语如 `"商品价格列表"` vs `"商品价格清单"` 极度敏感）；
    3. 英文/数字词元 Jaccard 相似度。
    """
    a = re.sub(r"[\s_\-:：,，。()（）\[\]]+", "", text_a.lower().strip())
    b = re.sub(r"[\s_\-:：,，。()（）\[\]]+", "", text_b.lower().strip())
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    # Strip common generic prefixes like 'v_', 'node_', 'vertex_' for comparison
    a_clean = re.sub(r"^(v_|vertex_|node_)", "", a)
    b_clean = re.sub(r"^(v_|vertex_|node_)", "", b)
    if a_clean and b_clean and a_clean == b_clean:
        return 0.98

    # Character 1-grams + 2-grams cosine similarity
    def _ngrams(s: str) -> Counter:
        grams = list(s) + [s[i : i + 2] for i in range(len(s) - 1)]
        return Counter(grams)

    vec_a = _ngrams(a_clean or a)
    vec_b = _ngrams(b_clean or b)
    common_keys = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    cosine = (dot / (norm_a * norm_b)) if (norm_a and norm_b) else 0.0

    # Substring containment bonus (e.g. "价格列表" in "价格列表与优惠")
    shorter, longer = (a_clean, b_clean) if len(a_clean) <= len(b_clean) else (b_clean, a_clean)
    containment = (len(shorter) / len(longer)) if (shorter and len(shorter) >= 2 and shorter in longer) else 0.0
    if shorter and len(shorter) >= 2 and shorter in longer:
        # Boost when one vertex name is a direct component of a composite downstream vertex name
        cosine = max(cosine, 0.55 + 0.35 * containment)

    return round(min(1.0, cosine), 4)


class CycleError(ValueError):
    """当插入的边导致有向图出现环路时抛出。"""


class DAG:
    """VEA V5 核心有向无环图容器。

    设计目标：让人类开发者和 LLM Agent 都能用最简单的原语搭建、合并、查看和局部修改工作流。
    """

    def __init__(self, name: str = "vea_v5_dag") -> None:
        self.name = name
        self.vertices: Dict[str, Vertex] = {}
        self.edges: Dict[str, Edge] = {}
        self._edge_seq: int = 0

    # =========================================================================
    # 1. 极简建图原语：添加顶点、添加边、添加 V-E-V 句子三元组
    # =========================================================================

    def add_vertex(
        self,
        name: str,
        content: Any = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Vertex:
        """插入或增量更新一个顶点（Patch 模式，绝不丢失已有数据或状态）。"""
        name = name.strip()
        if not name:
            raise ValueError("顶点名称不能为空")

        if name in self.vertices:
            v = self.vertices[name]
            if description:
                v.description = description
            if metadata:
                v.metadata.update(metadata)
            if content is not None and content != v.content:
                v.set_ready(content)
                self.invalidate_downstream(name)
            return v

        v = Vertex(
            name=name,
            content=content,
            description=description,
            status=VertexStatus.READY if content is not None else VertexStatus.PENDING,
            metadata=dict(metadata or {}),
        )
        self.vertices[name] = v
        return v

    def add_edge(
        self,
        source: str,
        target: str,
        action: str = "",
        edge_id: Optional[str] = None,
        kind: Optional[str] = None,
        prompt: Optional[str] = None,
        fn: Optional[Callable[[Any], Any]] = None,
        code: Optional[str] = None,
        merge_mode: Union[MergeMode, str] = MergeMode.OVERWRITE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Edge:
        """插入或更新一条计算边（`source -> target`）。

        - 如果 `source` 或 `target` 顶点还不存在，会自动创建；
        - 如果传入了 Python 函数 `fn` 或 `code`，自动将 `kind` 设为 `"code"`；
        - 插入后立即检测是否成环；若成环则自动回滚并抛出 `CycleError`；
        - 插入或修改成功后，自动将 `target` 及其下游顶点标记为 `pending` 以便增量重算。
        """
        source = source.strip()
        target = target.strip()
        if source == target:
            raise CycleError(f"不允许自环边: {source} -> {target}")

        self.add_vertex(source)
        self.add_vertex(target)

        if edge_id is None:
            self._edge_seq += 1
            edge_id = f"e_{self._edge_seq}_{source}_to_{target}"

        resolved_kind = kind
        if resolved_kind is None:
            if fn is not None or code is not None:
                resolved_kind = "code"
            else:
                resolved_kind = "llm"

        resolved_merge = MergeMode(merge_mode) if isinstance(merge_mode, str) else merge_mode

        new_edge = Edge(
            id=edge_id,
            source=source,
            target=target,
            action=action or prompt or edge_id,
            kind=resolved_kind,
            prompt=prompt or (action if resolved_kind == "llm" else None),
            fn=fn,
            code=code,
            merge_mode=resolved_merge,
            metadata=dict(metadata or {}),
        )

        old_edge = self.edges.get(edge_id)
        self.edges[edge_id] = new_edge

        # 成环保护：一旦检测到环路，立即回滚并报错
        cycle_path = self.find_cycle()
        if cycle_path:
            if old_edge is not None:
                self.edges[edge_id] = old_edge
            else:
                self.edges.pop(edge_id, None)
            raise CycleError(f"添加边 '{edge_id}' ({source} -> {target}) 会导致环路: {' -> '.join(cycle_path)}")

        # 增量失效：重置 target 及其下游顶点
        self.invalidate_from(target)
        return new_edge

    def add_virtual_edge(
        self,
        source: str,
        target: str,
        similarity: float = 1.0,
        reason: str = "相似顶点自动桥接",
        merge_mode: Union[MergeMode, str] = MergeMode.TEXT_JOIN,
        edge_id: Optional[str] = None,
    ) -> VirtualEdge:
        """在两个相似顶点之间插入一条虚拟边（`VirtualEdge`）。"""
        source = source.strip()
        target = target.strip()
        if source == target:
            raise CycleError(f"不允许在同一顶点上创建虚拟边: {source}")

        self.add_vertex(source)
        self.add_vertex(target)

        # 若已存在相同方向的边，直接返回或更新
        for existing in self.edges.values():
            if existing.source == source and existing.target == target:
                if isinstance(existing, VirtualEdge):
                    existing.similarity = similarity
                    existing.reason = reason
                    return existing

        if edge_id is None:
            self._edge_seq += 1
            edge_id = f"ve_{self._edge_seq}_{source}_sim_{target}"

        resolved_merge = MergeMode(merge_mode) if isinstance(merge_mode, str) else merge_mode
        v_edge = VirtualEdge(
            id=edge_id,
            source=source,
            target=target,
            action=f"≈虚拟边({similarity:.2f})",
            kind="virtual",
            merge_mode=resolved_merge,
            similarity=similarity,
            reason=reason,
        )

        self.edges[edge_id] = v_edge
        cycle_path = self.find_cycle()
        if cycle_path:
            self.edges.pop(edge_id, None)
            raise CycleError(f"虚拟边 '{edge_id}' ({source} -> {target}) 会导致环路: {' -> '.join(cycle_path)}")

        self.invalidate_from(target)
        return v_edge

    def add_step(
        self,
        source: str,
        action: Union[str, Callable[[Any], Any]],
        target: str,
        *,
        edge_id: Optional[str] = None,
        prompt: Optional[str] = None,
        code: Optional[str] = None,
        merge_mode: Union[MergeMode, str] = MergeMode.OVERWRITE,
        source_desc: str = "",
        target_desc: str = "",
    ) -> Edge:
        """最直观的一句话建图原语：添加“从 `source` 经过 `action` 到 `target`”的 `V-E-V` 步骤。

        示例：
            dag.add_step("x1", "提取关键数字", "z1")
            dag.add_step("z1", lambda x: int(x) * 2, "z2")
        """
        self.add_vertex(source, description=source_desc)
        self.add_vertex(target, description=target_desc)
        if callable(action):
            return self.add_edge(
                source=source,
                target=target,
                action=getattr(action, "__name__", "python_fn"),
                edge_id=edge_id,
                kind="code",
                fn=action,
                merge_mode=merge_mode,
            )
        return self.add_edge(
            source=source,
            target=target,
            action=str(action),
            edge_id=edge_id,
            kind="code" if code else "llm",
            prompt=prompt or str(action),
            code=code,
            merge_mode=merge_mode,
        )

    def add_triplet(self, triplet: Triplet) -> Edge:
        """将一个 `Triplet` (`V-E-V` 三元组对象) 插入到当前 DAG 中。"""
        return self.add_step(
            source=triplet.source_name,
            action=triplet.fn if triplet.fn is not None else triplet.edge_action,
            target=triplet.target_name,
            prompt=triplet.prompt,
            code=triplet.code,
            source_desc=triplet.source_desc,
            target_desc=triplet.target_desc,
        )

    def remove_edge(self, edge_id: str) -> bool:
        """删除一条边，并自动重置其原目标顶点及下游状态。"""
        edge = self.edges.pop(edge_id, None)
        if edge is None:
            return False
        self.invalidate_from(edge.target)
        return True

    def reconnect_edge(
        self,
        edge_id: str,
        new_source: Optional[str] = None,
        new_target: Optional[str] = None,
    ) -> Edge:
        """将已有边的起点或终点改接到新顶点上（自带成环检测与增量重置）。"""
        if edge_id not in self.edges:
            raise KeyError(f"边 '{edge_id}' 不存在")
        edge = self.edges[edge_id]
        old_source, old_target = edge.source, edge.target
        if new_source:
            self.add_vertex(new_source)
            edge.source = new_source.strip()
        if new_target:
            self.add_vertex(new_target)
            edge.target = new_target.strip()

        cycle_path = self.find_cycle()
        if cycle_path:
            edge.source, edge.target = old_source, old_target
            raise CycleError(f"重连边 '{edge_id}' 会导致环路: {' -> '.join(cycle_path)}")

        self.invalidate_from(old_target)
        self.invalidate_from(edge.target)
        return edge

    # =========================================================================
    # 2. 自动相似顶点虚拟边连接器 (Auto-Link Similar Vertices via VirtualEdge)
    # =========================================================================

    def auto_link_similar_vertices(
        self,
        threshold: float = 0.45,
        similarity_fn: Callable[[str, str], float] = default_text_similarity,
        merge_mode: MergeMode = MergeMode.TEXT_JOIN,
    ) -> List[VirtualEdge]:
        """自动扫描图中分立的 `V-E-V` 片段，在相似的“上游产出顶点”与“下游未连接输入顶点”之间架设虚拟边。

        核心逻辑：
        1. 找出所有作为某条边的输出（`out_candidates`：入度 > 0）的顶点；
        2. 找出所有作为某条边的输入但本身没有入边（`unfed_inputs`：入度 == 0 且无初始 `content`）的顶点；
        3. 计算每一对 `(u_out, v_in)` 的语义相似度 `sim = similarity_fn(u.semantic_text, v.semantic_text)`；
        4. 对于 `sim >= threshold` 且不会造成环路的顶点对，自动创建一条 `VirtualEdge(u_out -> v_in)`，
           注释写明“这两个顶点挺像的（相似度=sim）”，从而把分立的 `V-E-V` 句子粘合成一张完整连通的 DAG！
        """
        in_deg = self.in_degrees()
        out_deg = self.out_degrees()

        # 候选上游产出顶点：已经有入边的中间/结果顶点（或者已有初始数据的顶点）
        producers = [
            v for name, v in self.vertices.items()
            if in_deg[name] > 0 or v.is_ready
        ]
        # 候选待喂入顶点：有出边（被下游需要）、但自身既没有入边也没有初始数据的“悬空输入顶点”
        unfed_consumers = [
            v for name, v in self.vertices.items()
            if in_deg[name] == 0 and out_deg[name] > 0 and not v.is_ready
        ]

        # 按相似度从高到低配对连接
        candidate_pairs: List[Tuple[float, Vertex, Vertex]] = []
        for u in producers:
            for v in unfed_consumers:
                if u.name == v.name:
                    continue
                # 如果 v 可以到达 u，则连接 u -> v 必然成环，直接跳过
                if u.name in self.get_downstream_vertices(v.name):
                    continue
                sim = similarity_fn(u.semantic_text, v.semantic_text)
                if sim >= threshold:
                    candidate_pairs.append((sim, u, v))

        candidate_pairs.sort(key=lambda item: item[0], reverse=True)

        created_virtual_edges: List[VirtualEdge] = []
        for sim, u, v in candidate_pairs:
            try:
                ve = self.add_virtual_edge(
                    source=u.name,
                    target=v.name,
                    similarity=sim,
                    reason=f"顶点 '{u.name}' 与 '{v.name}' 语义相似 (score={sim:.2f})",
                    merge_mode=merge_mode,
                )
                created_virtual_edges.append(ve)
            except CycleError:
                continue

        return created_virtual_edges

    # =========================================================================
    # 3. 拓扑分析、起终点自动推导、环路检测与增量失效
    # =========================================================================

    def in_degrees(self) -> Dict[str, int]:
        deg = {name: 0 for name in self.vertices}
        for e in self.edges.values():
            deg[e.target] = deg.get(e.target, 0) + 1
        return deg

    def out_degrees(self) -> Dict[str, int]:
        deg = {name: 0 for name in self.vertices}
        for e in self.edges.values():
            deg[e.source] = deg.get(e.source, 0) + 1
        return deg

    def incoming_edges(self, vertex_name: str) -> List[Edge]:
        return [e for e in self.edges.values() if e.target == vertex_name]

    def outgoing_edges(self, vertex_name: str) -> List[Edge]:
        return [e for e in self.edges.values() if e.source == vertex_name]

    def roots(self) -> List[str]:
        """自动推导所有入口起点（入度为 0 的顶点）。"""
        in_deg = self.in_degrees()
        return [name for name, d in in_deg.items() if d == 0]

    def sinks(self) -> List[str]:
        """自动推导所有最终输出汇点（出度为 0 且入度 > 0 的顶点）。"""
        in_deg = self.in_degrees()
        out_deg = self.out_degrees()
        sinks_list = [name for name in self.vertices if out_deg[name] == 0 and in_deg[name] > 0]
        return sinks_list if sinks_list else list(self.vertices.keys())

    def get_downstream_vertices(self, start_vertex: str) -> List[str]:
        """利用 BFS 获取某个顶点下游所有受影响的顶点列表。"""
        visited: Set[str] = set()
        queue: List[str] = [ e.target for e in self.outgoing_edges(start_vertex) ]
        order: List[str] = []
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                order.append(curr)
                for e in self.outgoing_edges(curr):
                    if e.target not in visited:
                        queue.append(e.target)
        return order

    def invalidate_downstream(self, vertex_name: str) -> List[str]:
        """仅重置 `vertex_name` 的所有下游顶点为 `pending`（保留 `vertex_name` 自身数据）。"""
        downstream = self.get_downstream_vertices(vertex_name)
        for v_name in downstream:
            if v_name in self.vertices:
                self.vertices[v_name].reset()
        return downstream

    def invalidate_from(self, vertex_name: str) -> List[str]:
        """重置 `vertex_name` 自身及其所有下游顶点为 `pending`。"""
        if vertex_name in self.vertices:
            self.vertices[vertex_name].reset()
        downstream = self.invalidate_downstream(vertex_name)
        return [vertex_name] + downstream

    def find_cycle(self) -> Optional[List[str]]:
        """三色 DFS 检测图中是否存在环路；若存在则返回环路顶点路径列表，否则返回 None。"""
        color: Dict[str, int] = {name: 0 for name in self.vertices}  # 0: white, 1: gray, 2: black
        path: List[str] = []

        def dfs(u: str) -> Optional[List[str]]:
            color[u] = 1
            path.append(u)
            for e in self.outgoing_edges(u):
                v = e.target
                if color.get(v, 0) == 1:
                    idx = path.index(v)
                    return path[idx:] + [v]
                if color.get(v, 0) == 0:
                    res = dfs(v)
                    if res:
                        return res
            path.pop()
            color[u] = 2
            return None

        for name in sorted(self.vertices.keys()):
            if color[name] == 0:
                cyc = dfs(name)
                if cyc:
                    return cyc
        return None

    def topological_generations(self) -> List[List[str]]:
        """按 Kahn 算法返回 DAG 的分层拓扑序列（同一层的顶点可以完全并发执行）。"""
        cyc = self.find_cycle()
        if cyc:
            raise CycleError(f"图中存在环路，无法进行拓扑分层: {' -> '.join(cyc)}")

        in_deg = self.in_degrees()
        current_layer = sorted([n for n, d in in_deg.items() if d == 0])
        generations: List[List[str]] = []
        visited_count = 0

        while current_layer:
            generations.append(current_layer)
            visited_count += len(current_layer)
            next_layer: List[str] = []
            for u in current_layer:
                for e in self.outgoing_edges(u):
                    v = e.target
                    in_deg[v] -= 1
                    if in_deg[v] == 0:
                        next_layer.append(v)
            current_layer = sorted(next_layer)

        if visited_count < len(self.vertices):
            raise CycleError("图中存在未解析的环路结构")
        return generations

    # =========================================================================
    # 4. 可视化与序列化辅助方法
    # =========================================================================

    def visualize_ascii(self) -> str:
        """生成清晰直观的 ASCII 拓扑结构图（区分实线计算边与虚线相似虚拟边）。"""
        lines = [f"=== DAG 拓扑结构: {self.name} (顶点数={len(self.vertices)}, 边数={len(self.edges)}) ==="]
        try:
            layers = self.topological_generations()
            for idx, layer in enumerate(layers):
                v_strs = []
                for v_name in layer:
                    v = self.vertices[v_name]
                    tag = "✓READY" if v.is_ready else "…PENDING"
                    v_strs.append(f"[{v_name} ({tag})]")
                lines.append(f"  Tier {idx}: " + "  ".join(v_strs))
        except CycleError as err:
            lines.append(f"  [警告] {err}")

        lines.append("  边连接明细:")
        for e in self.edges.values():
            if e.is_virtual:
                lines.append(f"    ({e.source}) - -[{e.action}: { getattr(e, 'reason', '') }]- -> ({e.target})")
            else:
                lines.append(f"    ({e.source}) ===[{e.id} ({e.kind}): {e.action}]===> ({e.target})")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "roots": self.roots(),
            "sinks": self.sinks(),
            "vertices": {k: v.to_dict() for k, v in self.vertices.items()},
            "edges": {k: e.to_dict() for k, e in self.edges.items()},
        }
