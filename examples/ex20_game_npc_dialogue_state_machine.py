"""Example 20: Game NPC Dialogue & Affinity State Machine DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 20: Game NPC Dialogue State Machine DAG ===")
    dag = DAG("npc_dialogue_dag", models="models.json", llm_client=MockLLMClient())

    # Edge 0: Analyze player tone and attitude (Polite vs Hostile)
    def _evaluate_tone(utterance: str) -> str:
        if "请" in utterance or "help" in utterance.lower():
            return "tone:polite, affinity:+10"
        return "tone:neutral, affinity:0"

    dag.add_edge(
        input_tags=["player_utterance"],
        processor=_evaluate_tone,
        output_tags=["npc_affinity_delta"],
        layer=0,
    )

    # Edge 1: NPC response generation conditioned on affinity
    dag.add_edge(
        input_tags=["player_utterance", "npc_affinity_delta"],
        processor=lambda x: f"NPC_Aria: '旅行者，感谢你的礼貌。我这里有一张通往遗忘之谷的地图可以送给你。' [Affinity: {x[:30]}]",
        output_tags=["npc_response"],
        layer=1,
    )

    # Edge 2: Trigger Quest log update
    dag.add_edge(
        input_tags=["npc_response"],
        processor=lambda r: f"QuestLogUpdated: [Quest #14: 遗忘之谷的秘密] has been unlocked! [{r[:30]}]",
        output_tags=["quest_unlocked"],
        layer=2,
    )

    dag.add_content("你好，请问能否向您请教关于遗忘之谷的线索？", tags=["player_utterance"], gen=0)

    dag.run(until_tag="quest_unlocked", max_steps=5)
    quest = dag.get_result("quest_unlocked")
    print(f"Quest Result: {quest}")
    assert "QuestLogUpdated" in quest
    print("Example 20 Passed Successfully!")


if __name__ == "__main__":
    main()
