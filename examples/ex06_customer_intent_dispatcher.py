"""Example 06: Customer Support Multi-Intent Dispatcher DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 06: Customer Support Intent Dispatcher DAG ===")
    dag = DAG("intent_dispatcher_dag", models="models.json", llm_client=MockLLMClient())

    # Layer 0: Intent Classifier
    def _classify_intent(ticket: str) -> str:
        if "退款" in ticket or "扣费" in ticket or "refund" in ticket.lower():
            return "intent:billing"
        elif "报错" in ticket or "crash" in ticket.lower():
            return "intent:technical"
        return "intent:general"

    dag.add_edge(
        input_tags=["customer_ticket"],
        processor=_classify_intent,
        output_tags=["classified_intent"],
        layer=0,
    )

    # Layer 1: Handlers for different intents
    dag.add_edge(
        input_tags=["classified_intent"],
        processor=lambda intent: f"BillingSupportAgent: 已为您开启退款通道并进入人工审核 (基于 {intent})",
        output_tags=["support_resolution"],
        layer=1,
    )

    dag.add_content("我昨天被扣了双倍订阅费用，请协助退款！", tags=["customer_ticket"], gen=0)

    dag.run(until_tag="support_resolution", max_steps=4)
    res = dag.get_result("support_resolution")
    print(f"Customer Support Resolution: {res}")
    assert "BillingSupportAgent" in res
    print("Example 06 Passed Successfully!")


if __name__ == "__main__":
    main()
