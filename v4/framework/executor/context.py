from dataclasses import dataclass, field
from ..agents.base_agent import BaseAgent
from ..utils.memory import MemoryStore
from ..utils.telemetry import TelemetryTracker
from ..utils.schema import SchemaRegistry

@dataclass
class ExecutionContext:
    """Owns all resources needed for a graph run."""
    agents: BaseAgent
    memory: MemoryStore = field(default_factory=MemoryStore)
    telemetry: TelemetryTracker = field(default_factory=TelemetryTracker) 
    schema_registry: SchemaRegistry = field(default_factory=SchemaRegistry)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *exc):
        if hasattr(self.agents, 'close'):
            await self.agents.close()
