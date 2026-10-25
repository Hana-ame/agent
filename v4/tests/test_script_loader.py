"""Regression tests for framework/utils/script_loader.py.

Locks in the fix where ``load_class_from_script`` ignored the requested class
name (``default_class``) and returned the first subclass of ``base_class`` in
alphabetical ``inspect.getmembers`` order.

That made ``script.py:SummarizeEdge`` actually load ``FetchEdge`` (first Edge
subclass alphabetically) in MapEdge pipeline steps — every thread was fetched
twice and the structured title/summary post-processing never ran.
"""
import importlib.util
import os
import sys

from framework.edge import Edge
from framework.utils.script_loader import load_class_from_script

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1_EDGES = os.path.join(ROOT, "examples", "s1_ai_report_map", "s1_edges.py")


def test_load_class_by_name_returns_requested_class():
    cls = load_class_from_script(S1_EDGES, Edge, "SummarizeEdge")
    assert cls.__name__ == "SummarizeEdge"
    assert issubclass(cls, Edge)


def test_load_class_by_name_fetch_threads_edge():
    cls = load_class_from_script(S1_EDGES, Edge, "FetchThreadsEdge")
    assert cls.__name__ == "FetchThreadsEdge"


def test_load_class_by_name_is_not_alpha_first():
    # Regression: "SummarizeEdge" must NOT resolve to FetchEdge (first Edge
    # subclass alphabetically in this file) anymore.
    cls = load_class_from_script(S1_EDGES, Edge, "SummarizeEdge")
    assert cls.__name__ == "SummarizeEdge"
    assert cls.__name__ != "FetchEdge"


def test_load_class_autodiscover_returns_subclass():
    # No explicit name -> first Edge subclass in the file.
    cls = load_class_from_script(S1_EDGES, Edge, Edge)
    assert issubclass(cls, Edge)
    assert cls.__name__ in {
        "FetchEdge", "FetchThreadsEdge", "FilterEdge", "SelectEdge",
        "SummarizeEdge", "ProcessThreadsMap",
    }
