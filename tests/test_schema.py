import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pydantic import BaseModel, ValidationError
from framework import (
    Graph,
    Executor,
    MockAgent,
    SchemaRegistry,
    SchemaMismatchError
)

# 1. Define Pydantic Models
class UserProfile(BaseModel):
    user_id: int
    name: str

class Report(BaseModel):
    content: str
    confidence: float

# Register them
SchemaRegistry.register("UserProfile", UserProfile)
SchemaRegistry.register("Report", Report)


class TestStaticSchemaValidation:
    def test_schema_mismatch_raises_error(self):
        """Graph.validate() should raise SchemaMismatchError if schemas don't match."""
        config = {
            "vertices": [
                {"id": "A"},
                {"id": "B", "settings": {"input_schema": "UserProfile"}}
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "A",
                    "destination": "B",
                    "settings": {"output_schema": "Report"}  # Mismatch!
                }
            ]
        }
        with pytest.raises(ValueError) as excinfo:
            Graph.from_dict(config)
        assert "Schema Mismatch on edge 'e1'" in str(excinfo.value)
        assert "Edge outputs 'Report' but destination vertex 'B' expects 'UserProfile'" in str(excinfo.value)

    def test_schema_match_passes(self):
        """Graph.validate() should pass if schemas match."""
        config = {
            "vertices": [
                {"id": "A"},
                {"id": "B", "settings": {"input_schema": "UserProfile"}}
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "A",
                    "destination": "B",
                    "settings": {"output_schema": "UserProfile"}  # Match!
                }
            ]
        }
        g = Graph.from_dict(config)
        assert g is not None


class TestRuntimeSchemaValidation:
    @pytest.mark.asyncio
    async def test_runtime_validation_success(self):
        config = {
            "vertices": [{"id": "Start", "initial_data": [{"channel": "in", "value": "test"}]}, {"id": "End"}],
            "edges": [
                {
                    "id": "e1",
                    "source": "Start",
                    "destination": "End",
                    "channel": "in",
                    "settings": {"prompt": "generate profile", "output_schema": "UserProfile"}
                }
            ]
        }
        g = Graph.from_dict(config)
        
        # Agent correctly outputs the expected dict
        agent = MockAgent(response_fn=lambda d, p, m, s: {"user_id": 42, "name": "Alice"})
        executor = Executor(g, agents=agent)
        res = await executor.run()
        
        assert res.success is True
        data = await g.vertices["End"].fetch_data("in")
        assert data == {"user_id": 42, "name": "Alice"}

    @pytest.mark.asyncio
    async def test_runtime_validation_failure_triggers_retry(self):
        config = {
            "vertices": [{"id": "Start", "initial_data": [{"channel": "in", "value": "test"}]}, {"id": "End"}],
            "edges": [
                {
                    "id": "e1",
                    "source": "Start",
                    "destination": "End",
                    "channel": "in",
                    "settings": {
                        "output_schema": "UserProfile",
                        "retry_policy": {
                            "max_retries": 2,
                            "backoff_factor": 0.01,
                            "retry_on": ["ValidationError"]
                        }
                    }
                }
            ]
        }
        g = Graph.from_dict(config)
        
        call_count = 0
        def flawed_agent(d, p, m, s):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First time, output invalid data (missing 'user_id', wrong type for 'name')
                return {"name": 123}
            else:
                # Second time, correct it
                assert "[SYSTEM FEEDBACK: Your previous output produced a ValidationError" in p
                assert "user_id" in p
                return {"user_id": 99, "name": "Bob"}

        agent = MockAgent(response_fn=flawed_agent)
        executor = Executor(g, agents=agent)
        res = await executor.run()
        
        assert res.success is True
        assert call_count == 2
        data = await g.vertices["End"].fetch_data("in")
        assert data == {"user_id": 99, "name": "Bob"}
