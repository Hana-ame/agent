"""Meta-DAG (`meta_dag`): A DAG that modifies another DAG via JSON under the influence of a prompt.

Requirement:
- Input: a target DAG (`target_dag`) serialized as JSON + a modification `prompt`.
- Output: the modified DAG (`modified_dag`) serialized as JSON (and hydrated back into a `DAG` object).
- The Meta-DAG is itself a standard `DAG` instance (`models.json`, `contents.json`, `edges.json`).
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from dag_agent.engine import DAG
from dag_agent.llm import LLMClientProtocol
from dag_agent.schema import ContentItem, EdgeSpec


def _extract_json_object_with_span(text: str) -> Tuple[Optional[Dict[str, Any]], int, int]:
    """Extract the first valid JSON dictionary object and its `(start, end)` character span."""
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            try:
                obj, end = dec.raw_decode(text, i)
                if isinstance(obj, dict):
                    return obj, i, end
            except json.JSONDecodeError:
                pass
        i += 1
    return None, -1, -1


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    obj, _, _ = _extract_json_object_with_span(text)
    return obj


def apply_prompt_to_dag_dict(
    dag_dict: Dict[str, Any],
    prompt: str,
    llm_client: Optional[LLMClientProtocol] = None,
    models: Optional[Any] = None,
) -> Dict[str, Any]:
    """Apply natural-language or structured JSON instructions in `prompt` to `dag_dict`."""
    mutated = copy.deepcopy(dag_dict)
    mutated.setdefault("models", {})
    mutated.setdefault("contents", [])
    mutated.setdefault("edges", [])

    # 1. Check if prompt itself is a structured JSON patch or full replacement spec
    embedded_json = _extract_json_object(prompt)
    if embedded_json:
        if "edges" in embedded_json and "contents" in embedded_json:
            return embedded_json
        if any(
            k in embedded_json
            for k in ("add_edges", "remove_edge_ids", "add_contents", "add_models", "update_edges")
        ):
            if "add_edges" in embedded_json:
                for e in embedded_json["add_edges"]:
                    mutated["edges"].append(EdgeSpec.from_any(e).to_dict())
            if "remove_edge_ids" in embedded_json:
                remove_set = set(embedded_json["remove_edge_ids"])
                mutated["edges"] = [e for e in mutated["edges"] if e.get("id") not in remove_set]
            if "add_contents" in embedded_json:
                for c in embedded_json["add_contents"]:
                    mutated["contents"].append(ContentItem(c).to_dict())
            if "add_models" in embedded_json and isinstance(embedded_json["add_models"], dict):
                mutated["models"].update(embedded_json["add_models"])
            if "update_edges" in embedded_json and isinstance(embedded_json["update_edges"], dict):
                for eid, patch in embedded_json["update_edges"].items():
                    for e in mutated["edges"]:
                        if e.get("id") == eid:
                            e.update(patch)
            return mutated

    # 2. Natural-language directive parsing (supports Chinese & English modification instructions)
    lines = [ln.strip() for ln in re.split(r"[\n;；]+", prompt) if ln.strip()]
    applied_rule = False

    for line in lines:
        # Pattern A: Add edge, e.g. "添加边: [a, b] -> (model: prompt) -> [c] layer=1" or "从 [a] 经过 prompt 到 [b] layer=2"
        arrow_match = re.search(
            r"\[(?P<in>[^\]]+)\]\s*->\s*(?P<proc>.+?)\s*->\s*\[(?P<out>[^\]]+)\](?:\s*layer\s*=\s*(?P<layer>\d+))?",
            line,
        )
        cn_match = re.search(
            r"(?:添加边|新增边|增加边|从)\s*\[?(?P<in>[^\]\s]+)\]?\s*(?:经过|使用|通过)\s*(?P<proc>.+?)\s*(?:到|输出|产出)\s*\[?(?P<out>[^\]\s,，]+)\]?(?:.*?layer\s*[=:：]\s*(?P<layer>\d+))?",
            line,
        )
        m = arrow_match or cn_match
        if m:
            in_tags = [t.strip(" '\"") for t in m.group("in").split(",") if t.strip(" '\"")]
            out_tags = [t.strip(" '\"") for t in m.group("out").split(",") if t.strip(" '\"")]
            raw_proc = m.group("proc").strip()
            if raw_proc.startswith("(") and raw_proc.endswith(")"):
                raw_proc = raw_proc[1:-1].strip()
            layer_val = int(m.group("layer")) if m.group("layer") is not None else len(mutated["edges"])
            if ":" in raw_proc and not raw_proc.startswith("{"):
                mod_name, pr_text = raw_proc.split(":", 1)
                proc_spec: Any = {"model": mod_name.strip(), "prompt": pr_text.strip()}
            else:
                proc_spec = raw_proc
            new_edge = EdgeSpec(
                input_tags=in_tags,
                processor=proc_spec,
                output_tags=out_tags,
                layer=layer_val,
                count=0,
                id=f"edge_{len(mutated['edges'])}",
            )
            mutated["edges"].append(new_edge.to_dict())
            applied_rule = True
            continue

        # Pattern B: Delete edge by id or tag, e.g. "删除边 edge_1" or "remove edge edge_0"
        del_id_match = re.search(r"(?:删除边|移除边|delete edge|remove edge)\s+(?P<eid>edge_\w+|\w+)", line, re.I)
        if del_id_match:
            target = del_id_match.group("eid").strip()
            mutated["edges"] = [
                e
                for e in mutated["edges"]
                if e.get("id") != target and target not in e.get("output_tags", [])
            ]
            applied_rule = True
            continue

        # Pattern C: Add content item, e.g. "添加内容: tags=[raw] contents=Hello"
        content_match = re.search(
            r"(?:添加内容|新增内容|注入内容|add content)[:：\s]*tags=\[(?P<tags>[^\]]+)\]\s*contents=(?P<text>.+)$",
            line,
            re.I,
        )
        if content_match:
            tags = [t.strip(" '\"") for t in content_match.group("tags").split(",") if t.strip(" '\"")]
            text = content_match.group("text").strip().strip("'\"")
            mutated["contents"].append(ContentItem(tags=tags, contents=text, gen=0).to_dict())
            applied_rule = True
            continue

        # Pattern D: Add/update model, e.g. "添加模型 fast: endpoint=https://... apikey=sk-xxx"
        model_match = re.search(
            r"(?:添加模型|新增模型|add model)\s+(?P<key>[\w\-\.]+)[:：\s]+endpoint=(?P<ep>\S+)(?:\s+apikey=(?P<ak>\S+))?",
            line,
            re.I,
        )
        if model_match:
            k = model_match.group("key")
            ep = model_match.group("ep")
            ak = model_match.group("ak") or ""
            mutated["models"][k] = {"endpoint": ep, "apikey": ak, "model": k}
            applied_rule = True
            continue

        # Pattern E: Modify layer of an existing edge, e.g. "将 edge_0 的 layer 改为 5"
        layer_mod = re.search(r"(?:将|把)?\s*(?P<eid>edge_\d+)\s*的?\s*layer\s*(?:改为|设为|=)\s*(?P<lv>\d+)", line, re.I)
        if layer_mod:
            eid, lv = layer_mod.group("eid"), int(layer_mod.group("lv"))
            for e in mutated["edges"]:
                if e.get("id") == eid:
                    e["layer"] = lv
            applied_rule = True
            continue

    # 3. If no deterministic rule matched and an LLM client is provided, ask the LLM to transform the JSON
    if not applied_rule and llm_client is not None:
        meta_query = (
            "你是一个元DAG编辑器 (Meta-DAG Transformer)。请根据修改指令修改下面的 DAG JSON，"
            "保持合法的 `{models, contents, edges}` 结构并只输出修改后的 JSON：\n\n"
            f"【修改指令 Prompt】:\n{prompt}\n\n"
            f"【待修改 DAG JSON】:\n{json.dumps(mutated, ensure_ascii=False, indent=2)}"
        )
        model_cfg = models.resolve("default") if hasattr(models, "resolve") else None
        llm_resp = llm_client.complete(
            meta_query,
            model_config=model_cfg,
            phase="meta",
            is_clean_context=True,
        )
        parsed = _extract_json_object(llm_resp)
        if parsed and ("edges" in parsed or "contents" in parsed):
            return parsed

    return mutated


def create_meta_dag(
    *,
    llm_client: Optional[LLMClientProtocol] = None,
    models: Optional[Dict[str, Any]] = None,
) -> DAG:
    """Create the Meta-DAG (`meta_dag`), which is itself a standard `DAG`.

    Pipeline inside `meta_dag`:
    - Input items in `meta_dag.contents`:
      1. `{"tags": ["target_dag"], "contents": "<DAG JSON>", "gen": 0}`
      2. `{"tags": ["prompt"], "contents": "<Modification Instruction>", "gen": 0}`
    - Edge 0 (`layer=0`): `["target_dag", "prompt"]` -> parse & plan -> `["dag_edit_bundle"]`
      (Since `"target_dag"` and `"prompt"` start as two separate items, `meta_dag.step()`
      automatically uses `default_processor` N-in-1-out targeted assembly to combine them!)
    - Edge 1 (`layer=1`): `["dag_edit_bundle"]` -> apply modifications & validate schema -> `["modified_dag", "result"]`
    """
    meta = DAG("meta_dag", models=models, llm_client=llm_client)

    def _plan_modification(payload: Any, item: Any) -> str:
        raw_text = item.contents if isinstance(item, ContentItem) else str(payload)
        dag_obj, start, end = _extract_json_object_with_span(raw_text)
        if dag_obj is None:
            dag_obj = {"models": {}, "contents": [], "edges": []}
            prompt_part = raw_text.strip()
        else:
            prompt_part = (raw_text[:start] + "\n" + raw_text[end:]).strip()
        return json.dumps(
            {"target_dag": dag_obj, "prompt": prompt_part},
            ensure_ascii=False,
        )

    def _execute_and_validate(payload: Any, item: Any) -> str:
        raw_text = item.contents if isinstance(item, ContentItem) else str(payload)
        bundle = json.loads(raw_text)
        target_dag_dict = bundle["target_dag"]
        prompt_str = bundle["prompt"]
        modified_dict = apply_prompt_to_dag_dict(
            target_dag_dict,
            prompt_str,
            llm_client=meta.llm_client,
            models=meta.models,
        )
        validated_dag = DAG.from_dict(modified_dict, llm_client=meta.llm_client)
        return validated_dag.to_json(indent=2)

    meta.add_edge(
        input_tags=["target_dag", "prompt"],
        processor=_plan_modification,
        output_tags=["dag_edit_bundle"],
        layer=0,
    )
    meta.add_edge(
        input_tags=["dag_edit_bundle"],
        processor=_execute_and_validate,
        output_tags=["modified_dag", "result"],
        layer=1,
    )
    return meta


def transform_dag_with_meta_dag(
    target_dag: Union[DAG, Dict[str, Any], str],
    prompt: str,
    *,
    meta_dag: Optional[DAG] = None,
    llm_client: Optional[LLMClientProtocol] = None,
) -> DAG:
    """Run `meta_dag` on `(target_dag, prompt)` and return the modified `DAG` instance."""
    client = llm_client or (target_dag.llm_client if isinstance(target_dag, DAG) else None)
    meta = meta_dag or create_meta_dag(llm_client=client)

    if isinstance(target_dag, DAG):
        dag_json = target_dag.to_json(indent=2)
    elif isinstance(target_dag, dict):
        dag_json = json.dumps(target_dag, ensure_ascii=False, indent=2)
    else:
        dag_json = str(target_dag)

    meta.add_content(dag_json, tags=["target_dag"], gen=0)
    meta.add_content(prompt, tags=["prompt"], gen=0)
    meta.run(until_tag="result", max_steps=10)

    result_json = meta.get_result("result")
    if not result_json:
        raise RuntimeError("Meta-DAG did not produce a modified DAG under tag 'result'")
    return DAG.from_json(result_json, llm_client=client)
