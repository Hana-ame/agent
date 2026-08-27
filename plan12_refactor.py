import glob

def update_file(path, replacements):
    with open(path, 'r') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)

# 1. Rename PROCESSING -> AWAITING_EDGES
for path in glob.glob('framework/*.py') + glob.glob('tests/*.py'):
    update_file(path, [
        ('PROCESSING = "processing"', 'AWAITING_EDGES = "awaiting_edges"'),
        ('VertexState.PROCESSING', 'VertexState.AWAITING_EDGES'),
        ('state=processing', 'state=awaiting_edges'),
        ('VertexState.PROCESSING.value', 'VertexState.AWAITING_EDGES.value')
    ])

# 2. CQRS Refactor (Vertex)
with open('framework/vertex.py', 'r') as f:
    vertex_code = f.read()

# Remove READ from EdgeSignal
vertex_code = vertex_code.replace('READ = "read"\n    ', '')

# Replace handle_edge_signal signature
old_sig = """    async def handle_edge_signal(
        self,
        edge_id: str,
        signal: EdgeSignal,
        payload: Any = None,
        channel: str = "default",
    ):
        \"\"\"Handle state update or data requests from an edge.\"\"\"
        if signal == EdgeSignal.READ:
            async with self._lock:
                val = self._data_store.get(channel)
                logger.debug(
                    "[Vertex:%s] READ channel='%s' -> %s",
                    self.id, channel, repr(val)[:120],
                )
                return val"""

new_sig = """    async def fetch_data(self, channel: str = "default") -> Any:
        \"\"\"Command: Fetch data from the vertex's data store.\"\"\"
        async with self._lock:
            val = self._data_store.get(channel)
            logger.debug(f"[Vertex:{self.id}] FETCH channel='{channel}' -> {repr(val)[:120]}")
            return val

    async def receive_signal(
        self,
        edge_id: str,
        signal: EdgeSignal,
        payload: Any = None,
        channel: str = "default",
    ):
        \"\"\"Event: Receive state update or completed payload from an edge.\"\"\""""

vertex_code = vertex_code.replace(old_sig, new_sig)
# There is a missing `elif signal == EdgeSignal.COMPLETED:` which should become `if signal == EdgeSignal.COMPLETED:`
vertex_code = vertex_code.replace('elif signal == EdgeSignal.COMPLETED:', 'if signal == EdgeSignal.COMPLETED:')
with open('framework/vertex.py', 'w') as f:
    f.write(vertex_code)

# 3. CQRS Refactor (Edge)
update_file('framework/edge.py', [
    ('data = await source_vertex.handle_edge_signal(self.id, EdgeSignal.READ, channel=self.channel)', 'data = await source_vertex.fetch_data(channel=self.channel)'),
    ('await dest_vertex.handle_edge_signal(self.id, EdgeSignal.COMPLETED, payload=result, channel=self.channel)', 'await dest_vertex.receive_signal(self.id, EdgeSignal.COMPLETED, payload=result, channel=self.channel)'),
    ('await dest_vertex.handle_edge_signal(self.id, EdgeSignal.ABORTED)', 'await dest_vertex.receive_signal(self.id, EdgeSignal.ABORTED)'),
    ('await dest_vertex.handle_edge_signal(self.id, EdgeSignal.FAILED, payload=str(exc))', 'await dest_vertex.receive_signal(self.id, EdgeSignal.FAILED, payload=str(exc))')
])

# 4. CQRS Refactor (Tests & executor)
for path in glob.glob('tests/*.py') + ['framework/executor.py']:
    update_file(path, [
        ('.handle_edge_signal("", EdgeSignal.READ, channel=', '.fetch_data(channel='),
        ('.handle_edge_signal(self.id, EdgeSignal.READ, channel=', '.fetch_data(channel='),
        ('.handle_edge_signal("", EdgeSignal.COMPLETED,', '.receive_signal("", EdgeSignal.COMPLETED,'),
        ('.handle_edge_signal("edge", EdgeSignal.COMPLETED,', '.receive_signal("edge", EdgeSignal.COMPLETED,'),
        ('.handle_edge_signal("edge", EdgeSignal.ABORTED)', '.receive_signal("edge", EdgeSignal.ABORTED)'),
        ('.handle_edge_signal("edge", EdgeSignal.FAILED,', '.receive_signal("edge", EdgeSignal.FAILED,')
    ])

print("Plan 1 & 2 complete")
