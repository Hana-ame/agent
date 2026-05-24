from .base import BaseOrganelle, OrganelleContext, ExecutionResult, create_organelle


class OptimizerOrganelle(BaseOrganelle):
    def __init__(self, context: OrganelleContext):
        super().__init__(context)
        self._inner = create_organelle(context)

    def execute(self, protein: str, task_input: dict) -> ExecutionResult:
        return self._inner.execute(protein, task_input)

    def estimate_quality(self, result: ExecutionResult, task_input: dict) -> float:
        base_score = super().estimate_quality(result, task_input)
        opt_markers = ["optimized", "faster", "efficient", "reduced", "improved", "better"]
        matches = sum(1 for m in opt_markers if m in result.output.lower())
        bonus = min(0.3, matches * 0.05)
        return min(1.0, base_score + bonus)
