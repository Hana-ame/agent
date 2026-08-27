import re

with open('tests/test_vertex.py', 'r') as f:
    content = f.read()

# Replace .get(data_id, tags)
content = re.sub(
    r'\.get\("([^"]+)"(?:,\s*(\[[^\]]+\]))?\)',
    lambda m: f'.handle_edge_signal("", EdgeSignal.READ, data_id="{m.group(1)}"' + (f', tags={m.group(2)}' if m.group(2) else '') + ')',
    content
)

# Replace .set(data, data_id, tags, edge_id)
# There are variations. Let's do it carefully.
content = re.sub(
    r'\.set\("([^"]+)",\s*"([^"]+)"(?:,\s*(\[[^\]]+\]))?(?:,\s*edge_id="([^"]+)")?\)',
    lambda m: f'.handle_edge_signal("{m.group(4) or ""}", EdgeSignal.COMPLETED, payload="{m.group(1)}", data_id="{m.group(2)}"' + (f', tags={m.group(3)}' if m.group(3) else '') + ')',
    content
)

with open('tests/test_vertex.py', 'w') as f:
    f.write(content)

