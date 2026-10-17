"""Example 16: Interactive Story Narrative Branching & Convergence DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 16: Interactive Story Branching DAG ===")
    dag = DAG("story_branching_dag", models="models.json", llm_client=MockLLMClient())

    # Branch A: Explore ancient ruins
    dag.add_edge(
        input_tags=["hero_prologue"],
        processor=lambda p: "BranchRuins: The hero discovers an ancient glowing artifact in the sunken temple.",
        output_tags=["ruins_arc"],
        layer=0,
    )

    # Branch B: Journey through enchanted forest
    dag.add_edge(
        input_tags=["hero_prologue"],
        processor=lambda p: "BranchForest: The hero befriends a wise elder dragon guarding the mystic glade.",
        output_tags=["forest_arc"],
        layer=0,
    )

    # Convergence: Climax finale combining both arcs
    dag.add_edge(
        input_tags=["ruins_arc", "forest_arc"],
        processor=lambda arcs: f"ClimaxFinale: With the glowing artifact and dragon ally united, the kingdom is saved! Arcs: {arcs[:50]}",
        output_tags=["story_epilogue"],
        layer=1,
    )

    dag.add_content("Chapter 1: The young traveler embarks from the valley town at dawn.", tags=["hero_prologue"], gen=0)

    dag.run(until_tag="story_epilogue", max_steps=5)
    epilogue = dag.get_result("story_epilogue")
    print(f"Story Epilogue: {epilogue[:100]}...")
    assert "ClimaxFinale" in epilogue
    print("Example 16 Passed Successfully!")


if __name__ == "__main__":
    main()
