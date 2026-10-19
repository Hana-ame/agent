# Examples

This directory contains **5 executable examples** for VEA V6.

| Example | Description | Execution Command |
|---|---|---|
| `05_vea_v6_quickstart.py` | Direct V6 DAG creation and execution with preset-edge auto-synthesis | `python3 examples/05_vea_v6_quickstart.py` |
| `06_vea_v6_meta_builder.py` | Meta-DAG builder: a V6 DAG that builds another V6 DAG | `python3 examples/06_vea_v6_meta_builder.py` |
| `07_vea_v6_tag_reverse_openai.py` | Fetch content by tag (`todo`) → reverse it → write back to `dag.contents`; plus an OpenAI-compatible edge via the model registry | `python3 examples/07_vea_v6_tag_reverse_openai.py` |
| `08_vea_v6_openai_edge_configs.py` | Three ways to configure an OpenAI-compatible edge: **registry** / **inline** / **mixed** (inline overrides registry) | `python3 examples/08_vea_v6_openai_edge_configs.py` |
| `09_vea_v6_bootstrap.py` | **Bootstrap**: a DAG that generates a complete DAG framework from a description, and proves the generated framework can regenerate itself byte-for-byte | `python3 examples/09_vea_v6_bootstrap.py` |

## Running

```bash
python3 examples/05_vea_v6_quickstart.py
python3 examples/06_vea_v6_meta_builder.py
python3 examples/07_vea_v6_tag_reverse_openai.py
python3 examples/08_vea_v6_openai_edge_configs.py
python3 examples/09_vea_v6_bootstrap.py
```

`08` runs offline by default and only inspects the resolved configuration.
To hit a real endpoint:

```bash
VEA_REAL=1 python3 examples/08_vea_v6_openai_edge_configs.py
```

## Tests

```bash
python3 tests/test_vea_v6_all.py          # 18 suites
python3 -m unittest discover -s tests -v
python3 -m pytest tests/test_vea_v6_all.py -v
```

## Docs

See [docs/ARCHITECTURE_V6.md](../docs/ARCHITECTURE_V6.md) for the full architecture guide.
