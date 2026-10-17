"""Example 10: Natural Language to SQL Generator & Safety Validator DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 10: NL2SQL & Safety Validator DAG ===")
    dag = DAG("nl2sql_dag", models="models.json", llm_client=MockLLMClient())

    # Edge 0: Generate Draft SQL
    dag.add_edge(
        input_tags=["user_query"],
        processor=lambda q: f"SELECT user_id, SUM(amount) AS total FROM orders WHERE status = 'completed' GROUP BY user_id; -- query: {q[:30]}",
        output_tags=["draft_sql"],
        layer=0,
    )

    # Edge 1: Read-Only & AST Safety Check
    def _validate_sql(sql: str) -> str:
        prohibited = ["drop", "delete", "truncate", "update", "insert"]
        if any(w in sql.lower() for w in prohibited):
            return "REJECTED: Destructive query detected"
        return f"VALIDATED: Read-only query approved. SQL: {sql.strip()}"

    dag.add_edge(
        input_tags=["draft_sql"],
        processor=_validate_sql,
        output_tags=["safe_sql_statement"],
        layer=1,
    )

    dag.add_content("统计已完成订单中每位用户的消费总额", tags=["user_query"], gen=0)

    dag.run(until_tag="safe_sql_statement", max_steps=4)
    res = dag.get_result("safe_sql_statement")
    print(f"Validated SQL: {res}")
    assert "VALIDATED" in res
    print("Example 10 Passed Successfully!")


if __name__ == "__main__":
    main()
