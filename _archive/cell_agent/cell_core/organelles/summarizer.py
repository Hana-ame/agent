from .base import BaseOrganelle, OrganelleContext, ExecutionResult, create_organelle


class SummarizerOrganelle(BaseOrganelle):
    def __init__(self, context: OrganelleContext):
        super().__init__(context)
        self._inner = create_organelle(context)

    def execute(self, protein: str, task_input: dict) -> ExecutionResult:
        return self._inner.execute(protein, task_input)

    def estimate_quality(self, result: ExecutionResult, task_input: dict) -> float:
        base_score = super().estimate_quality(result, task_input)
        input_len = len(task_input.get("text", ""))
        output_len = len(result.output)
        if input_len > 0 and output_len < input_len * 0.8:
            return min(1.0, base_score + 0.2)
        return base_score * 0.6
