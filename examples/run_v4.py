#!/usr/bin/env python3
"""Run a V4 manifest end to end: load, seed SQLite, execute.

``examples/run.py`` is the legacy **V1** runner and is left untouched. This file
is the V4 counterpart and is intentionally a thin wrapper around
``framework.run_v4`` so the installed ``vea-run-v4`` console script and this
script can never drift apart.

Usage:
    python examples/run_v4.py <path/to/graph.json> [--session ID] [--db DB]
                              [--script-root DIR] [--json] [--quiet]

Examples:
    python examples/run_v4.py examples/custom_edge/config.json --session demo
    vea-run-v4 examples/custom_edge/config.json --session demo   # installed form
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from framework.run_v4 import main

if __name__ == "__main__":
    sys.exit(main())
