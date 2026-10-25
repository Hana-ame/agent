"""Dashboard static assets (HTML/JS/CSS) served by :mod:`framework.server`.

This module exists so that setuptools treats the directory as a package and
includes the non-Python assets in the built distribution (see
``[tool.setuptools.package-data]`` in ``pyproject.toml``). Without it a
``pip install`` would serve an empty dashboard.
"""
