from .base import BaseOrganelle, OrganelleContext, ExecutionResult, create_organelle


class TranslatorOrganelle(BaseOrganelle):
    def __init__(self, context: OrganelleContext):
        super().__init__(context)
        self._inner = create_organelle(context)

    def execute(self, protein: str, task_input: dict) -> ExecutionResult:
        return self._inner.execute(protein, task_input)

    def estimate_quality(self, result: ExecutionResult, task_input: dict) -> float:
        base_score = super().estimate_quality(result, task_input)
        if len(result.output) > len(task_input.get("text", "")) * 0.3:
            return base_score
        return base_score * 0.5
