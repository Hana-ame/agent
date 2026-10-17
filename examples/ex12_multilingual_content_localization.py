"""Example 12: Multilingual Content Localization & Cultural Adaptation DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 12: Multilingual Localization DAG ===")
    dag = DAG("localization_dag", models="models.json", llm_client=MockLLMClient())

    # Branch 1 (English localization)
    dag.add_edge(
        input_tags=["master_copy"],
        processor=lambda t: f"EN_Local: 'Unleash next-gen autonomous coding with frictionless speed.' [From: {t[:20]}]",
        output_tags=["en_marketing_copy"],
        layer=0,
    )

    # Branch 2 (Japanese localization with polite keigo)
    dag.add_edge(
        input_tags=["master_copy"],
        processor=lambda t: f"JA_Local: '次世代自律コーディングAIが、開発効率を飛躍的に向上させます。' [From: {t[:20]}]",
        output_tags=["ja_marketing_copy"],
        layer=0,
    )

    # Convergence (Multi-Language Bundle)
    dag.add_edge(
        input_tags=["en_marketing_copy", "ja_marketing_copy"],
        processor=lambda x: f"GlobalCampaignBundle: EN and JA copies ready for worldwide launch. Details: {x[:60]}",
        output_tags=["global_campaign_bundle"],
        layer=1,
    )

    dag.add_content("全新AI编程智能体，解放开发者生产力，极速交付高质量系统", tags=["master_copy"], gen=0)

    dag.run(until_tag="global_campaign_bundle", max_steps=5)
    bundle = dag.get_result("global_campaign_bundle")
    print(f"Global Campaign: {bundle[:100]}...")
    assert "GlobalCampaignBundle" in bundle
    print("Example 12 Passed Successfully!")


if __name__ == "__main__":
    main()
