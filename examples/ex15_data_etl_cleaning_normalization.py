"""Example 15: Data ETL Pipeline with Cleaning & Normalization DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 15: Data ETL Pipeline DAG ===")
    dag = DAG("etl_pipeline_dag", models="models.json", llm_client=MockLLMClient())

    # Step 1: Strip leading/trailing whitespaces and invalid chars
    dag.add_edge(
        input_tags=["raw_csv_record"],
        processor=lambda raw: raw.strip().replace("  ", " "),
        output_tags=["trimmed_record"],
        layer=0,
    )

    # Step 2: Timestamp format normalization (e.g. YYYY/MM/DD -> ISO 8601)
    dag.add_edge(
        input_tags=["trimmed_record"],
        processor=lambda t: f"NormalizedRecord(id=101, timestamp=2026-10-17T00:00:00Z, val=98.5) from {t}",
        output_tags=["normalized_record"],
        layer=1,
    )

    # Step 3: Schema validation & load readiness
    dag.add_edge(
        input_tags=["normalized_record"],
        processor=lambda n: f"WarehouseLoadReady: Validated schema [PASS]. Ready for ClickHouse ingestion. {n[:40]}",
        output_tags=["warehouse_ready"],
        layer=2,
    )

    dag.add_content("  101, 2026/10/17,   98.50  ", tags=["raw_csv_record"], gen=0)

    dag.run(until_tag="warehouse_ready", max_steps=5)
    ready = dag.get_result("warehouse_ready")
    print(f"ETL Result: {ready}")
    assert "WarehouseLoadReady" in ready
    print("Example 15 Passed Successfully!")


if __name__ == "__main__":
    main()
