# Examples

This directory contains **2 executable examples** for VEA V6.

| Example | Description | Execution Command |
|---|---|---|
| `05_vea_v6_quickstart.py` | Direct V6 DAG creation and execution with preset-edge auto-synthesis | `python3 examples/05_vea_v6_quickstart.py` |
| `06_vea_v6_meta_builder.py` | Meta-DAG builder: a V6 DAG that builds another V6 DAG | `python3 examples/06_vea_v6_meta_builder.py` |

## Running

```bash
python3 examples/05_vea_v6_quickstart.py
python3 examples/06_vea_v6_meta_builder.py
```

## Tests

```bash
python3 tests/test_vea_v6_all.py
python3 -m pytest tests/test_vea_v6_all.py -v
```

## Docs

See [docs/ARCHITECTURE_V6.md](../docs/ARCHITECTURE_V6.md) for the full architecture guide.
