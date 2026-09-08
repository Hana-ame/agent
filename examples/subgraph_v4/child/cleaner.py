"""Text cleaner transform for child subgraph."""

def clean_text(content: str) -> str:
    """Normalize whitespace and convert to title case."""
    normalized = ' '.join(content.strip().split())
    return normalized.title()
