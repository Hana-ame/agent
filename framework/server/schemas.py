"""Pydantic schemas and request models for VEA v4 Server."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from framework.vertex_v4 import VertexAttributeV4, VertexStateV4


class VertexCreateOrUpdateRequest(BaseModel):
    name: str = Field(..., description="Unique vertex name in session")
    content: str = Field(default="", description="Payload content")
    attributes: List[str] = Field(default_factory=list, description="Vertex attributes")
    state: str = Field(default=VertexStateV4.IDLE.value, description="Lifecycle state")
    processed_count: int = Field(default=0, description="Processing counter")

    @field_validator('state')
    @classmethod
    def validate_state(cls, v: str) -> str:
        valid_states = {s.value for s in VertexStateV4}
        if v not in valid_states:
            raise ValueError(f"Invalid state: {v}")
        return v

    @field_validator('attributes')
    @classmethod
    def validate_attributes(cls, v: List[str]) -> List[str]:
        valid_attrs = {a.value for a in VertexAttributeV4}
        for attr in v:
            if attr not in valid_attrs:
                raise ValueError(f"Invalid attribute: {attr}")
        return v


class EdgeCreateOrUpdateRequest(BaseModel):
    id: str = Field(..., description="Unique edge identifier")
    type: str = Field(default="code", description="Edge type: code, llm, or reflexive")
    input_vertex: str = Field(..., description="Input vertex name")
    output_vertex: str = Field(..., description="Output vertex name")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Edge settings")
    script: Optional[str] = Field(default=None, description="Script path or callable specification")
    trigger_state: Optional[str] = Field(default=None, description="Trigger state for reflexive edge")
    target_state: Optional[str] = Field(default=None, description="Target reset state for reflexive edge")
    max_retries: Optional[int] = Field(default=None, description="Max retries for reflexive edge")


class WorkflowRunRequest(BaseModel):
    max_concurrency: int = Field(default=4, ge=1, le=64, description="Max concurrent edges")
    timeout: float = Field(default=120.0, gt=0.0, description="Execution timeout in seconds")


class VertexReentryRequest(BaseModel):
    new_content: Optional[str] = Field(default=None, description="Optional new input content")
    reset_state: str = Field(default=VertexStateV4.TODO.value, description="Target state for affected downstream nodes")
    clear_content: bool = Field(default=False, description="Whether to clear content of reset downstream nodes")


class EdgeReconnectRequest(BaseModel):
    new_input_vertex: Optional[str] = Field(default=None, description="New input vertex name")
    new_output_vertex: Optional[str] = Field(default=None, description="New output vertex name")


class SubgraphSpliceRequest(BaseModel):
    target_vertex: str = Field(..., description="Target vertex name to replace with subgraph")
    subgraph_manifest: Optional[str] = Field(default=None, description="Path to subgraph manifest JSON file")
    subgraph_data: Optional[Dict[str, Any]] = Field(default=None, description="In-memory subgraph definition")
    name_prefix: Optional[str] = Field(default=None, description="Prefix for inserted subgraph entities")
    entry_vertex: Optional[str] = Field(default=None, description="Explicit entry vertex of subgraph")
    exit_vertex: Optional[str] = Field(default=None, description="Explicit exit vertex of subgraph")


class SubgraphInsertRequest(BaseModel):
    subgraph_manifest: Optional[str] = Field(default=None, description="Path to subgraph manifest JSON file")
    subgraph_data: Optional[Dict[str, Any]] = Field(default=None, description="In-memory subgraph definition")
    incoming_bindings: Optional[Dict[str, str]] = Field(default=None, description="{parent_vertex: sub_entry}")
    outgoing_bindings: Optional[Dict[str, str]] = Field(default=None, description="{sub_exit: parent_vertex}")
    name_prefix: Optional[str] = Field(default=None, description="Prefix for inserted subgraph entities")


class SubgraphAddRequest(BaseModel):
    subgraph_manifest: Optional[str] = Field(default=None, description="Path to subgraph manifest JSON file")
    subgraph_data: Optional[Dict[str, Any]] = Field(default=None, description="In-memory subgraph definition")
    name_prefix: Optional[str] = Field(default=None, description="Prefix for added subgraph entities")
    connections: Optional[List[Dict[str, Any]]] = Field(default=None, description="Explicit connections between parent and subgraph")
    incoming_bindings: Optional[Dict[str, str]] = Field(default=None, description="{parent_vertex: sub_entry}")
    outgoing_bindings: Optional[Dict[str, str]] = Field(default=None, description="{sub_exit: parent_vertex}")
    source: Optional[str] = Field(default=None, description="Provenance source identifier for loaded nodes")


class RouteAndRunRequest(BaseModel):
    task: str = Field(..., description="User task description or query")
    tool_id: Optional[str] = Field(default=None, description="Explicit tool override. If None, classifier is used.")
    use_llm: bool = Field(default=True, description="Whether to use LLM for semantic intent routing")
    max_concurrency: int = Field(default=4, ge=1, le=64, description="Max concurrent edges")
    timeout: float = Field(default=120.0, gt=0.0, description="Execution timeout in seconds")
