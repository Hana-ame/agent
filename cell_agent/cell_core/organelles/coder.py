from .base import BaseOrganelle, OrganelleContext, ExecutionResult, create_organelle


class CoderOrganelle(BaseOrganelle):
    def __init__(self, context: OrganelleContext):
        super().__init__(context)
        self._inner = create_organelle(context)

    def execute(self, protein: str, task_input: dict) -> ExecutionResult:
        return self._inner.execute(protein, task_input)

    def estimate_quality(self, result: ExecutionResult, task_input: dict) -> float:
        score = super().estimate_quality(result, task_input)
        code_keywords = ["def ", "class ", "import ", "function", "return", "```"]
        matches = sum(1 for kw in code_keywords if kw in result.output)
        bonus = min(0.3, matches * 0.05)
        return min(1.0, score + bonus)
