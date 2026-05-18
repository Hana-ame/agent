import time
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class OrganelleContext:
    organelle_id: int
    name: str
    capability: str
    model: str
    api_base: str
    cost_per_token: float


@dataclass
class ExecutionResult:
    output: str
    cost: float = 0.0
    time_ms: int = 0
    quality_score: Optional[float] = None
    error: Optional[str] = None
    token_count: int = 0


class BaseOrganelle(ABC):
    def __init__(self, context: OrganelleContext):
        self.context = context

    @abstractmethod
    def execute(self, protein: str, task_input: dict) -> ExecutionResult:
        ...

    def estimate_quality(self, result: ExecutionResult, task_input: dict) -> float:
        if result.error:
            return 0.0
        length = len(result.output)
        if length < 10:
            return 0.2
        return min(1.0, length / 2000)

    def run(self, protein: str, task_input: dict) -> ExecutionResult:
        t0 = time.time()
        result = self.execute(protein, task_input)
        result.time_ms = int((time.time() - t0) * 1000)
        if result.quality_score is None:
            result.quality_score = self.estimate_quality(result, task_input)
        return result


class MockLocalOrganelle(BaseOrganelle):
    def execute(self, protein: str, task_input: dict) -> ExecutionResult:
        simulated = (
            f"# Simulated output for '{self.context.name}'\n"
            f"## Protein used:\n{protein}\n\n"
            f"## Result:\n"
            f"[{self.context.capability}] processed task: {task_input.get('task', 'N/A')}\n"
            f"Generated at: {datetime.now().isoformat()}\n"
            f"{'=' * 40}\n"
            f"Mock output content for {self.context.name} agent.\n"
        )
        return ExecutionResult(
            output=simulated,
            cost=0.0,
            token_count=len(simulated) // 4,
        )


class HTTPOrganelle(BaseOrganelle):
    def __init__(self, context: OrganelleContext, session=None):
        super().__init__(context)
        self._session = session

    def _build_payload(self, protein: str) -> dict:
        if "openai" in self.context.api_base.lower() or "v1" in self.context.api_base:
            return {
                "model": self.context.model,
                "messages": [{"role": "user", "content": protein}],
                "temperature": 0.3,
            }
        return {
            "model": self.context.model,
            "prompt": protein,
            "temperature": 0.3,
        }

    def execute(self, protein: str, task_input: dict) -> ExecutionResult:
        import httpx

        payload = self._build_payload(protein)
        headers = {"Content-Type": "application/json"}

        try:
            resp = httpx.post(
                self.context.api_base,
                json=payload,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_output = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                or data.get("response", "")
            )
            usage = data.get("usage", {})
            token_count = usage.get("total_tokens", len(raw_output) // 4)
            cost = token_count * self.context.cost_per_token
            return ExecutionResult(
                output=raw_output,
                cost=cost,
                token_count=token_count,
            )
        except Exception as e:
            return ExecutionResult(
                output="",
                cost=0.0,
                error=f"HTTP call failed: {e}",
            )


def create_organelle(context: OrganelleContext) -> BaseOrganelle:
    model = context.model.lower()
    if model in ("local", "mock"):
        return MockLocalOrganelle(context)
    if context.api_base:
        return HTTPOrganelle(context)
    return MockLocalOrganelle(context)
