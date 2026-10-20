import sys, json
sys.path.insert(0, r"C:\Users\lumin\Downloads\agent")
import vea_v6.parts.bootstrap as B
from vea_v6.parts.dag import DAG

# 冒烟隔离：自复现深校验与本次「DAG 接线」改动无关，打桩避免重 introspection 干扰
B.verify_self_reproduction = lambda src: (True, "smoke-stub")

base = dict(B.framework_sources())

def fake_ask(prompt):
    # spec 边
    if "框架现状" in prompt or "写可执行规格" in prompt:
        return json.dumps(B.FRAMEWORK_SPEC, ensure_ascii=False)
    # skeleton 边
    if "框架文件与公开签名" in prompt:
        return json.dumps({"files": []}, ensure_ascii=False)
    # 逐文件边：返回「已去掉 ``` 的真实源码」，既过契约闸门又绕开 extractor 误截断
    for p, src in base.items():
        if p in prompt:
            return src.replace("```", "")
    return ""

dag = B.create_generation_dag(llm=object())   # llm 仅占位；真实 ask 已被下面覆盖
dag.ask_sync = lambda prompt, model=None: fake_ask(prompt)

# ---- 结构校验 ----
edges = dag.edges
print("edge count:", len(edges))
file_tags = ["code_" + p.replace("/", "__").replace(".", "_")
             for p in B.FRAME_FILES if p != B.MAIN_SOURCE]
present = [t for e in edges for t in e.output_tags]
missing = [ft for ft in file_tags if ft not in present]
print("expected file tags:", file_tags)
print("missing file tags:", missing)
assert not missing, f"缺单文件节点: {missing}"
for need in ("skeleton_code", "frame_code", "manifest", "result"):
    assert need in present, f"缺 {need}"
layers = {}
for e in edges:
    layers.setdefault(e.layer, []).append(e.output_tags)
print("layers:", {k: [t for e in v for t in e] for k, v in sorted(layers.items())})

# ---- 执行校验 ----
dag.send({"tags": ["input"], "content": "smoke test description", "gen": 0})
ticks = 0
while not dag.has_result() and ticks < 8:
    dag.run()
    ticks += 1
assert dag.has_result(), "DAG 未产出 tag='result'"
res = dag.get_result()
print("ticks:", ticks, "has_result:", dag.has_result(), "result len:", len(res or ""))
fc = dag.find_by_tag("frame_code")
print("frame_code items:", len(fc))
print("LAST_VERDICT keys:", list(B._LAST_VERDICT.keys()))
print("rejected:", B._LAST_VERDICT.get("rejected"))
print("SMOKE OK")
