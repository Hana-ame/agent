from typing import List, Optional

def process_items(items: List[str]) -> Optional[str]:
    if items:
        return items[0].upper()
    return None

result = process_items(['apple', 'banana'])
print(result)
