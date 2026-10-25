from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v9.infer import InferenceEngine  # noqa: E402
from v9.knowledge import Knowledge  # noqa: E402


def _gen(payload):
    return lambda prompt: json.dumps(payload, ensure_ascii=False)


def _chain():
    k = Knowledge()
    k.add_entity("A")
    k.add_entity("B")
    k.add_entity("C")
    c1 = k.add_claim("A", "uses", "B", "A 使用 B", source_url="https://x/1")
    c2 = k.add_claim("B", "part_of", "C", "B 属于 C", source_url="https://x/2")
    return k, c1, c2


def test_derive_chain_marks_derived_from():
    k, c1, c2 = _chain()
    engine = InferenceEngine(k, generate=_gen({
        "relation": "uses", "statement": "A 使用 C", "confidence": 0.7}),
        fetch=lambda e: {})
    res = engine.derive()
    assert res["status"] == "claim"
    claim = k.claims()[-1]
    assert claim["source_url"] is None
    assert json.loads(claim["derived_from"]) == [c1, c2]
    k.close()


def test_derive_none_does_not_write():
    k, _, _ = _chain()
    engine = InferenceEngine(k, generate=_gen({"relation": "none"}),
                             fetch=lambda e: {})
    res = engine.derive()
    assert res["status"] == "no_relation"
    assert len(k.claims()) == 2
    k.close()


def test_derive_no_chain():
    k = Knowledge()
    k.add_entity("A")
    k.add_entity("B")
    k.add_claim("A", "uses", "B", "A 使用 B", source_url="https://x")
    engine = InferenceEngine(k, generate=_gen({"relation": "uses"}),
                             fetch=lambda e: {})
    assert engine.derive()["status"] == "no_chain"
    k.close()
