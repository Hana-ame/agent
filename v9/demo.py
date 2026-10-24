"""真实演示：v8 负责抓取/生成，v7 图当生成本体，v9 负责知识增长。

    python -m v9.demo
"""
from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from v9.executor import V8Executor  # noqa: E402
from v9.infer import InferenceEngine  # noqa: E402
from v9.knowledge import Knowledge  # noqa: E402

TITLES = ["Directed_acyclic_graph", "Topological_sorting", "Blackboard_system"]


def wiki_url(title: str) -> str:
    return ("https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(title.replace(" ", "_")))


def main() -> int:
    k = Knowledge(ROOT / "v9" / "data" / "knowledge.sqlite3")
    for title in TITLES:
        name = title.replace("_", " ")
        k.add_entity(name, source_url=wiki_url(name))

    ex = V8Executor()
    engine = InferenceEngine(
        k, generate=ex.generate,
        fetch=lambda name: ex.fetch(wiki_url(name)))
    for res in engine.run(cycles=3):
        print(res)
    print("stats", k.stats())
    for c in k.claims():
        print(f"- {c['subject']} --{c['predicate']}--> {c['object']} "
              f"({c['confidence']}) {c['source_url']}")
    k.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
