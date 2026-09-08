"""Exceptions for GraphV4 topology and validation."""

class GraphTopologyError(Exception):
    """Raised when graph definition contains circular dependencies or broken links."""
    pass
