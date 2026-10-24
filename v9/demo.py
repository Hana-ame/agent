"""真实演示：Wikipedia 抓证据 + 模型推理，把新关系写回知识库。

    python -m v9.demo
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v8"))

from dag8 import PiAgent, PiConfig  # noqa: E402
from v9.infer import InferenceEngine  # noqa: E402
from v9.knowledge import Knowledge  # noqa: E402

TITLES = ["Directed_acyclic_graph", "Topological_sorting", "Blackboard_system"]


def wiki(entity: str) -> dict:
    title = urllib.parse.quote(entity.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    req = urllib.request.Request(url, headers={"User-Agent": "v9-demo/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {"url": url, "text": data.get("extract") or data.get("description") or ""}


def main() -> int:
    k = Knowledge(ROOT / "v9" / "data" / "knowledge.sqlite3")
    for t in TITLES:
        k.add_entity(t.replace("_", " "), source_url=wiki(t)["url"])

    agent = PiAgent(PiConfig(provider="laptop", model="deepseek-v4-flash",
                             tools=False, timeout=180, retries=2))
    engine = InferenceEngine(k, generate=lambda p: agent.run(p), fetch=wiki)
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
