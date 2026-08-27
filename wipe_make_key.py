import re

with open("framework/vertex.py", "r") as f:
    code = f.read()

# Completely remove _make_key method
code = re.sub(r'    def _make_key\(.*?\n        return \([^\)]+\)\n', '', code, flags=re.DOTALL)
# Replace all self._make_key(channel) with str(channel)
code = code.replace("self._make_key(channel)", "str(channel)")

with open("framework/vertex.py", "w") as f:
    f.write(code)
