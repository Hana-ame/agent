import json
import urllib.request
import asyncio
import logging
from framework.edge import Edge
from framework.vertex import EdgeSignal

logger = logging.getLogger("llm_edge")

class RealLLMEdge(Edge):
    async def execute(self, source_vertex, dest_vertex, pi_agent):
        """Override execute to bypass the mock pi_agent and hit a real LLM."""
        logger.info(f"[RealLLMEdge:{self.id}] Intercepted execution to hit opencode.ai API")
        try:
            # 1. Read source data
            data = await source_vertex.handle_edge_signal(self.id, EdgeSignal.READ, data_id=self.data_id, tags=self.tags)
            if data is None:
                raise ValueError(f"No data received from source vertex '{self.source_id}'.")

            # 2. Build the request
            url = "https://opencode.ai/zen/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer public",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            # Use the model specified in the edge configuration, default to hy3-free
            target_model = self.model if self.model and self.model != "default" else "hy3-free"
            
            payload = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": str(data)}
                ]
            }

            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode('utf-8'), 
                headers=headers, 
                method="POST"
            )

            # 3. Call the API asynchronously
            def fetch():
                with urllib.request.urlopen(req) as response:
                    return json.loads(response.read().decode('utf-8'))

            logger.info(f"[RealLLMEdge:{self.id}] Calling {url} with model {target_model}...")
            response_data = await asyncio.to_thread(fetch)
            
            # 4. Parse the result
            result = response_data['choices'][0]['message']['content']
            logger.info(f"[RealLLMEdge:{self.id}] Received response: {repr(result)[:50]}...")

            # 5. Write to destination
            await dest_vertex.handle_edge_signal(self.id, EdgeSignal.COMPLETED, payload=result, data_id=self.data_id, tags=self.tags)
            self.completed = True
            self.result = result
            return result

        except Exception as exc:
            self.error = str(exc)
            logger.error(f"[RealLLMEdge:{self.id}] FAILED: {exc}", exc_info=True)
            # Propagate error to destination vertex to prevent deadlocks
            await dest_vertex.handle_edge_signal(self.id, EdgeSignal.FAILED, payload=str(exc))
            raise
