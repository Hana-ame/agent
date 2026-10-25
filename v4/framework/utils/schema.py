import logging
from typing import Dict, Type, Any, Optional

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = object
    ValidationError = Exception
    PYDANTIC_AVAILABLE = False

logger = logging.getLogger("vertex_edge_agent.schema")

class SchemaRegistry:
    """Global registry for Pydantic schemas to enable Type-Safe data flow."""
    _schemas: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, schema_class: Type[BaseModel]):
        if not PYDANTIC_AVAILABLE:
            logger.warning("[SchemaRegistry] Pydantic is not installed. Schema validation will be disabled.")
            return
        if not issubclass(schema_class, BaseModel):
            raise TypeError(f"Schema '{name}' must be a subclass of pydantic.BaseModel")
        cls._schemas[name] = schema_class
        logger.debug("[SchemaRegistry] Registered schema: '%s'", name)

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseModel]]:
        return cls._schemas.get(name)

    @classmethod
    def clear(cls):
        cls._schemas.clear()

class SchemaMismatchError(ValueError):
    """Raised during graph validation if connected schemas do not match."""
    pass
