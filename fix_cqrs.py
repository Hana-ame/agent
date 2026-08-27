with open("framework/vertex.py", "r") as f:
    code = f.read()

import re

# We need to replace the entire `handle_edge_signal` signature and body.
old_func = re.search(r'    async def handle_edge_signal\(.*?def get_all_data', code, flags=re.DOTALL)

new_func = """    async def fetch_data(self, channel: str = "default") -> Any:
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
    ) -> Any:
        \"\"\"Event: Receive state update or completed payload from an edge.\"\"\"
        if signal == EdgeSignal.COMPLETED:
            data = payload
            key = self._make_key(channel)
            logger.debug("[Vertex:%s] COMPLETED %s <- %s", self.id, key, repr(data)[:120])

            # --- run vertex script on_receive hook ---
            if hasattr(self, "on_receive") and callable(getattr(self, "on_receive")):
                try:
                    data = self.on_receive(data, channel, self.settings)
                    logger.debug("[Vertex:%s] self.on_receive returned: %s", self.id, repr(data)[:120])
                except Exception as exc:
                    logger.warning("[Vertex:%s] self.on_receive REJECTED data: %s", self.id, exc)
                    raise DataRejectedError(f"Vertex '{self.id}' rejected data: {exc}") from exc
            elif self._script_module and hasattr(self._script_module, "on_receive"):
                try:
                    data = self._script_module.on_receive(
                        data, channel, self.settings
                    )
                    logger.debug(
                        "[Vertex:%s] on_receive returned: %s", self.id, repr(data)[:120]
                    )
                except Exception as exc:
                    logger.warning(
                        "[Vertex:%s] on_receive REJECTED data: %s", self.id, exc
                    )
                    raise DataRejectedError(
                        f"Vertex '{self.id}' rejected data: {exc}"
                    ) from exc

            async with self._lock:
                self._data_store[key] = data
                if edge_id:
                    self.completed_incoming_edges.add(edge_id)
                else:
                    self._received_input_count += 1
                
                total = len(self.incoming_edges) if self.incoming_edges else self.required_input_count
                logger.info(
                    "[Vertex:%s] Input received (completed: %d, aborted: %d, total: %d)",
                    self.id, len(self.completed_incoming_edges), len(self.aborted_incoming_edges), total
                )
                
                # Check readiness based on incoming edge settlement
                is_ready = False
                if self.incoming_edges:
                    total_settled = len(self.completed_incoming_edges) + len(self.aborted_incoming_edges)
                    is_ready = total_settled >= len(self.incoming_edges) and len(self.completed_incoming_edges) > 0
                elif self.required_input_count > 0:
                    is_ready = self._received_input_count >= self.required_input_count

                if is_ready:
                    self.state = VertexState.READY
            return True

        elif signal == EdgeSignal.ABORTED:
            logger.debug("[Vertex:%s] ABORTED signal from edge '%s'", self.id, edge_id)
            async with self._lock:
                self.aborted_incoming_edges.add(edge_id)
                
                total = len(self.incoming_edges) if self.incoming_edges else self.required_input_count
                total_settled = len(self.completed_incoming_edges) + len(self.aborted_incoming_edges)
                
                if total > 0 and total_settled >= total:
                    if len(self.completed_incoming_edges) > 0:
                        self.state = VertexState.READY
                    else:
                        self.abort_reason = payload or f"All {total} incoming edges aborted"
                        self.state = VertexState.ABORTED
            return True

        elif signal == EdgeSignal.FAILED:
            logger.error("[Vertex:%s] FAILED signal from edge '%s': %s", self.id, edge_id, payload)
            async with self._lock:
                self.error_message = str(payload)
                self.state = VertexState.ERROR
            return True

    async def get_all_data"""

if old_func:
    code = code[:old_func.start()] + new_func + code[old_func.end()-18:]
else:
    print("Could not find handle_edge_signal!")

with open("framework/vertex.py", "w") as f:
    f.write(code)

