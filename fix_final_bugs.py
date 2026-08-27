import re
with open("framework/executor.py", "r") as f:
    code = f.read()
code = code.replace('f"{k[0]}:{','.join(k[1])}": val for k, val in data.items()', 'f"{k}": val for k, val in data.items()')
with open("framework/executor.py", "w") as f:
    f.write(code)

with open("tests/test_vertex.py", "r") as f:
    code = f.read()
code = code.replace('handle_edge_signal', 'receive_signal')
with open("tests/test_vertex.py", "w") as f:
    f.write(code)
