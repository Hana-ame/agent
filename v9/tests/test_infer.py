from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v9.infer import InferenceEngine  # noqa: E402
from v9.knowledge import Knowledge  # noqa: E402


class _Fetch:
    def __init__(self):
        self.data = {
            "a": {"url": "https://x/a", "text": "A 的资料"},
            "b": {"url": "https://x/b", "text": "B 的资料"},
            "c": {"url": "https://x/c", "text": "C 的资料"},
        }

    def __call__(self, entity):
        return self.data[entity]


def _gen(payload):
    return lambda prompt: json.dumps(payload, ensure_ascii=False)


def test_claim_from_external_source():
    k = Knowledge()
    k.add_entity("a")
    k.add_entity("b")
    engine = InferenceEngine(k, generate=_gen({
        "relation": "rel", "statement": "A 与 B 有关系",
        "confidence": 0.8, "evidence_url": "https://x/a"}), fetch=_Fetch())
    res = engine.step()
    assert res["status"] == "claim"
    claims = k.claims()
    assert len(claims) == 1
    assert claims[0]["source_url"] == "https://x/a"
    assert k.get_source("https://x/a") == "A 的资料"
    k.close()


def test_rejects_source_not_fetched():
    k = Knowledge()
    k.add_entity("a")
    k.add_entity("b")
    engine = InferenceEngine(k, generate=_gen({
        "relation": "rel", "statement": "编的",
        "confidence": 0.9, "evidence_url": "https://evil"}), fetch=_Fetch())
    res = engine.step()
    assert res["status"] == "rejected"
    assert k.claims() == []
    k.close()


def test_none_marks_explored_and_moves_on():
    k = Knowledge()
    for name in ("a", "b", "c"):
        k.add_entity(name)
    engine = InferenceEngine(k, generate=_gen({"relation": "none"}),
                             fetch=_Fetch())
    assert engine.step()["status"] == "no_relation"
    pairs = k.unexplored_pairs(10)
    assert ("a", "c") in pairs
    assert ("a", "b") not in pairs
    k.close()


def test_rejects_overlong_statement():
    k = Knowledge()
    k.add_entity("a")
    k.add_entity("b")
    engine = InferenceEngine(k, generate=_gen({
        "relation": "rel", "statement": "x" * 201,
        "confidence": 0.9, "evidence_url": "https://x/a"}), fetch=_Fetch())
    res = engine.step()
    assert res["status"] == "rejected"
    assert k.claims() == []
    k.close()


def test_accepts_canonical_wikipedia_url():
    k = Knowledge()
    k.add_entity("a")
    k.add_entity("b")

    def fetch(entity):
        return {"url": f"https://en.wikipedia.org/api/rest_v1/page/summary/{entity}",
                "text": f"{entity} text"}

    engine = InferenceEngine(k, generate=_gen({
        "relation": "rel", "statement": "A 与 B 有关系",
        "confidence": 0.8,
        "evidence_url": "https://en.wikipedia.org/wiki/a"}), fetch=fetch)
    res = engine.step()
    assert res["status"] == "claim"
    assert k.claims()[0]["source_url"].endswith("/summary/a")
    k.close()
