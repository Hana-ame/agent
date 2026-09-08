"""Text enricher transform for child subgraph."""
import json

def enrich_text(content: str) -> str:
    """Enrich cleaned text with word count and metadata."""
    words = content.split()
    result = {
        'processed_text': content,
        'word_count': len(words),
        'status': 'enriched_by_child_subgraph'
    }
    return json.dumps(result, ensure_ascii=False)
