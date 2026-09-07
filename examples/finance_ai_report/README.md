# Finance AI Report (MapEdge) — Local Execution

> Documented following the "Problem / Solution / Changes / Verification" format: demonstrates a finance topic report experiment adapting the MapEdge architecture (fetch -> LLM filter for finance/macro topics -> ProcessThreadsMap concurrent fetch and summarize -> `report.md`).

## Issue 1: Validating End-to-End Pipeline on Financial Topics

### Problem
Existing report generators focused on general AI topics; crawling, filtering, and summarizing financial and macroeconomic topics required verification against different forum sections and board layouts.

### Solution
Clone the MapEdge architecture from `s1_ai_report_map`, configure filtering keywords for finance and macroeconomics, and isolate report aggregation.

### Changes
- `examples/finance_ai_report/config.json`: MapEdge configuration with domain filtering.
- `examples/finance_ai_report/finance_edges.py`: MapEdge pipeline with fetch and summarize steps.
- `examples/finance_ai_report/vertex/report_hook.py`: Report accumulator hook.
- `examples/finance_ai_report/demo.py`: Execution demo script.
- `tests/test_s1_edges.py`: Added `finance_edges.py` to test paths (10 tests passed).

### Verification
- **Test Plan**: Financial thread ingestion -> LLM topic filtering -> MapEdge concurrent summarization -> `report.md`.
- **Method**: `python examples/finance_ai_report/demo.py` (with proxy/endpoint in config).
- **Result**: `report.md` generated successfully; `pytest tests/test_s1_edges.py -q` = **10 passed**.

## Known Limitations
- MapEdge 24h window excludes older thread starter posts; direct routes capture full history.
- Filtering and summarization prompts and unbounded list handling (see `s1_ai_report_map/README.md` Issue 2).

## Files

- `config.json`, `demo.py`, `finance_edges.py`, `vertex/report_hook.py`, `report.md`
