"""VEA V5 “第一个 VEA DAG”：分立句子自动拆解 V-E-V 与相似顶点虚拟边合并器 (Merger DAG).

本模块实现了用户提出的核心构想：
> “创建多个‘从 x1 经过 y1 到 z1’、‘从 z1 经过 y2 到 z2’这样一句一句分立的句子，能够自动合并成 DAG。
>  这个合并器本身就是第一个 VEA DAG：输入文字，然后一层一层输出分立的 V-E-V，
>  然后相似的 V 之间用虚拟 E 连接起来（就说这两个挺像的），最后构成能够运行的 DAG。”

合并器本身作为一个标准的 3 层 VEA V5 DAG (`meta_dag`) 运行：
- **Tier 0 起点 (`v_raw_text`)**：接收用户或 LLM 写出的一段或多句分立的自然语言描述；
- **Tier 1 第一层边 (`e_extract_vev` -> `v_vev_list`)**：把自然语言逐句拆解为独立的 `V - E - V` 三元组列表；
- **Tier 2 第二层边 (`e_find_similar_vertices` -> `v_virtual_links`)**：对比所有分立 `V-E-V` 的产出顶点与输入顶点，
  发现名称相同则直接合一，发现语义相似（“这两个挺像的”）则自动生成虚拟边 `VirtualEdge`；
- **Tier 3 第三层边 (`e_build_executable_dag` -> `v_final_dag`)**：将所有实线计算边与虚线虚拟边组装成一张无环、可执行的业务 `DAG`。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from vea_v5.dag import DAG, default_text_similarity
from vea_v5.executor import Executor
from vea_v5.llm import LLMProtocol
from vea_v5.primitives import MergeMode, Triplet, VirtualEdge


# 支持多种中英文“从 X 经过 Y 到 Z”句式的正则表达式集合
_SENTENCE_PATTERNS = [
    # 1. 标准中文句式：从 X 经过/通过/利用/执行/做 Y 到/得到/生成/产出/输出 Z
    re.compile(
        r"^(?:创建|添加|步骤\d*[:：.\s]*)?(?:从|由|把|将)\s*(?P<src>[^\s,，。;；]+?)\s*"
        r"(?:经过|通过|经由|利用|使用|执行|进行|做)\s*(?P<act>.+?)\s*"
        r"(?:到|至|得到|生成|产出|输出为|输出|变为|转化为)\s*(?P<dst>[^\s,，。;；]+?)$"
    ),
    # 2. 箭头三段式：X --[Y]--> Z 或 X -[Y]-> Z 或 X ==(Y)==> Z
    re.compile(
        r"^(?P<src>[^\-\=\>]+?)\s*[\-\=]+\[\s*(?P<act>.+?)\s*\][\-\=]*\>\s*(?P<dst>.+?)$"
    ),
    # 3. 简单双箭头式：X -> Y -> Z
    re.compile(
        r"^(?P<src>[^\-\>]+?)\s*\-\>\s*(?P<act>[^\-\>]+?)\s*\-\>\s*(?P<dst>[^\-\>]+?)$"
    ),
]


def parse_vev_from_text(text: Union[str, Sequence[str]]) -> List[Triplet]:
    """从多行文本或句子列表中提取分立的 `V - E - V` 三元组 (`Triplet`)。

    支持同句连续链式写法，例如：
    `"从 x1 经过 y1 到 z1 从 z1 经过 y2 到 z2"`
    会自动切分为两句并提取出：
    - `Triplet(x1 --[y1]--> z1)`
    - `Triplet(z1 --[y2]--> z2)`
    """
    if isinstance(text, str):
        # 将按行、分号、句号或连续出现的“从 ... 经过 ...”拆分为独立子句
        normalized = re.sub(r"(?<=[a-zA-Z0-9_\u4e00-\u9fa5])\s+(?=从\s+[^\s]+\s+(?:经过|通过|经由))", "\n", text)
        raw_lines = re.split(r"[\n;；。]+", normalized)
    else:
        raw_lines = []
        for item in text:
            normalized = re.sub(r"(?<=[a-zA-Z0-9_\u4e00-\u9fa5])\s+(?=从\s+[^\s]+\s+(?:经过|通过|经由))", "\n", str(item))
            raw_lines.extend(re.split(r"[\n;；。]+", normalized))

    triplets: List[Triplet] = []
    for line in raw_lines:
        clean = line.strip()
        # 去掉开头的序号如 "1. "、"1、"、"- "
        clean = re.sub(r"^(?:\d+[\.\、\)\s]+|[\-\*\•]\s+)", "", clean).strip()
        if not clean:
            continue

        matched = False
        for pat in _SENTENCE_PATTERNS:
            m = pat.match(clean)
            if m:
                src = m.group("src").strip(" '\"\t")
                act = m.group("act").strip(" '\"\t")
                dst = m.group("dst").strip(" '\"\t")
                triplets.append(
                    Triplet(
                        source_name=src,
                        edge_action=act,
                        target_name=dst,
                        raw_sentence=clean,
                    )
                )
                matched = True
                break

        if not matched:
            raise ValueError(
                f"无法从句子中解析出 V-E-V 三元组: '{clean}'。"
                "请使用类似 '从 x1 经过 y1 到 z1' 或 'x1 --[y1]--> z1' 的格式。"
            )

    return triplets


@dataclass
class MergerResult:
    """“第一个 VEA DAG（合并器）”的运行产出结果。"""

    compiled_dag: DAG
    extracted_triplets: List[Triplet] = field(default_factory=list)
    virtual_edges: List[VirtualEdge] = field(default_factory=list)
    similarity_logs: List[str] = field(default_factory=list)
    meta_dag_trace: Optional[Any] = None


class SentenceToDAGMerger:
    """“第一个 VEA DAG”：将分立的 V-E-V 句子自动对齐相似顶点并编译为可执行 DAG 的元图合并器。

    该合并器内部本身就是一个拥有 3 层计算边的 VEA V5 `DAG` (`self.meta_dag`)：
        v_raw_sentences
             │
             ▼ [e_1_extract_vev] (第1层：逐句解析出分立的 V-E-V 三元组)
        v_discrete_vevs
             │
             ▼ [e_2_align_similar_vertices] (第2层：计算顶点相似度，识别“这两个挺像的”候选对)
        v_aligned_plan
             │
             ▼ [e_3_assemble_dag] (第3层：插入实线边与虚拟边 VirtualEdge，生成可运行的业务 DAG)
        v_executable_dag
    """

    def __init__(
        self,
        similarity_threshold: float = 0.45,
        similarity_fn: Callable[[str, str], float] = default_text_similarity,
        default_merge_mode: MergeMode = MergeMode.TEXT_JOIN,
        edge_bindings: Optional[Dict[str, Union[Callable[[Any], Any], str]]] = None,
    ) -> None:
        """初始化合并器。

        Args:
            similarity_threshold: 判定两个不同名顶点“挺像的”并自动架设虚拟边的相似度阈值（默认 0.45）。
            similarity_fn: 顶点语义相似度计算函数。
            default_merge_mode: 虚拟边汇入同一点时的默认数据合并方式。
            edge_bindings: 可选的动作绑定字典 `{edge_action: python_fn_or_prompt}`，
                           用于在编译出的业务 DAG 中为特定动作赋予具体函数或提示词。
        """
        self.similarity_threshold = similarity_threshold
        self.similarity_fn = similarity_fn
        self.default_merge_mode = default_merge_mode
        self.edge_bindings = dict(edge_bindings or {})
        self.meta_dag = self._build_meta_dag()

    def _build_meta_dag(self) -> DAG:
        """构建合并器自身的 3 层 VEA V5 元图 (`Meta_Merger_DAG`)。"""
        dag = DAG(name="Meta_Merger_DAG")

        # 第 1 层：输入原始分立句子 -> 输出分立的 V-E-V 三元组列表
        def _layer1_extract(raw_input: Union[str, Sequence[str]]) -> List[Triplet]:
            return parse_vev_from_text(raw_input)

        # 第 2 层：输入分立的 V-E-V 列表 -> 分析哪些不同名的顶点“挺像的”，生成虚拟边连接计划
        def _layer2_match_similar(triplets: List[Triplet]) -> Dict[str, Any]:
            # 统计所有产出顶点（Z）和所有输入顶点（X）
            produced_names = {t.target_name for t in triplets}
            consumed_names = {t.source_name for t in triplets}

            # 精确同名匹配的顶点会自动合一；重点找“被下游消费、但没有同名上游产出”的悬空输入顶点
            unfed_inputs = sorted(consumed_names - produced_names)
            producers = sorted(produced_names)

            proposed_virtual_links: List[Dict[str, Any]] = []
            logs: List[str] = []

            for p_name in producers:
                for c_name in unfed_inputs:
                    if p_name == c_name:
                        continue
                    sim = self.similarity_fn(p_name, c_name)
                    if sim >= self.similarity_threshold:
                        msg = f"发现相似顶点: '{p_name}' 与 '{c_name}' 挺像的 (相似度={sim:.2f}) -> 建立虚拟边 [{p_name} - -> {c_name}]"
                        logs.append(msg)
                        proposed_virtual_links.append(
                            {
                                "source": p_name,
                                "target": c_name,
                                "similarity": sim,
                                "reason": f"'{p_name}' 与 '{c_name}' 挺像的 (相似度={sim:.2f})",
                            }
                        )

            return {
                "triplets": triplets,
                "virtual_links": proposed_virtual_links,
                "logs": logs,
            }

        # 第 3 层：将分立 V-E-V 三元组与虚拟边计划组装成最终可运行的业务 DAG
        def _layer3_compile(plan: Dict[str, Any]) -> MergerResult:
            triplets: List[Triplet] = plan["triplets"]
            virtual_links: List[Dict[str, Any]] = plan["virtual_links"]
            logs: List[str] = plan["logs"]

            compiled = DAG(name="Merged_Executable_DAG")

            # 1. 插入所有分立的 V-E-V 三元组（同名顶点如 z1 会自动合并为同一个 Vertex）
            for idx, t in enumerate(triplets, start=1):
                bound = self.edge_bindings.get(t.edge_action)
                if callable(bound):
                    compiled.add_step(
                        t.source_name,
                        bound,
                        t.target_name,
                        edge_id=f"e_{idx}_{t.edge_action}",
                    )
                elif isinstance(bound, str):
                    compiled.add_step(
                        t.source_name,
                        t.edge_action,
                        t.target_name,
                        edge_id=f"e_{idx}_{t.edge_action}",
                        prompt=bound,
                    )
                else:
                    compiled.add_triplet(t)

            # 2. 插入所有连接相似顶点的虚拟边 (VirtualEdge)
            created_ves: List[VirtualEdge] = []
            for vlink in virtual_links:
                # 避免连接后产生环路
                if vlink["source"] in compiled.get_downstream_vertices(vlink["target"]):
                    continue
                ve = compiled.add_virtual_edge(
                    source=vlink["source"],
                    target=vlink["target"],
                    similarity=vlink["similarity"],
                    reason=vlink["reason"],
                    merge_mode=self.default_merge_mode,
                )
                created_ves.append(ve)

            return MergerResult(
                compiled_dag=compiled,
                extracted_triplets=triplets,
                virtual_edges=created_ves,
                similarity_logs=logs,
            )

        dag.add_step("v_raw_sentences", _layer1_extract, "v_discrete_vevs", edge_id="e_1_extract_vev")
        dag.add_step("v_discrete_vevs", _layer2_match_similar, "v_aligned_plan", edge_id="e_2_align_similar_vertices")
        dag.add_step("v_aligned_plan", _layer3_compile, "v_executable_dag", edge_id="e_3_compile_dag")
        return dag

    def merge(self, sentences: Union[str, Sequence[str]]) -> MergerResult:
        """运行“第一个 VEA DAG（合并器）”，将分立句子编译为可执行的业务 DAG。"""
        exec_engine = Executor(self.meta_dag)
        # 每次调用前重置元图起点，触发三层流水线执行
        self.meta_dag.add_vertex("v_raw_sentences", content=sentences)
        self.meta_dag.invalidate_downstream("v_raw_sentences")
        report = exec_engine.run_sync()
        if not report.success:
            raise RuntimeError(f"合并器 DAG 执行失败: {report.errors}")
        result: MergerResult = report.final_output
        result.meta_dag_trace = report
        return result
