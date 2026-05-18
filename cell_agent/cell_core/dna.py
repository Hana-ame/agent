import json
import time
from typing import Optional

from .db import get_connection
from .ribosome import Ribosome
from .organelles.base import OrganelleContext, ExecutionResult
from .organelles.coder import CoderOrganelle
from .organelles.translator import TranslatorOrganelle
from .organelles.summarizer import SummarizerOrganelle
from .organelles.reviewer import ReviewerOrganelle
from .organelles.optimizer import OptimizerOrganelle

ORGANELLE_MAP = {
    "coder": CoderOrganelle,
    "translator": TranslatorOrganelle,
    "summarizer": SummarizerOrganelle,
    "reviewer": ReviewerOrganelle,
    "optimizer": OptimizerOrganelle,
}


class DNAExecutor:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.ribosome = Ribosome()

    def run_task(self, task_id: int) -> dict:
        conn = get_connection(self.db_path)
        cur = conn.cursor()

        cur.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        task = dict(cur.fetchone())
        dna_id = task["dna_id"]
        input_json = json.loads(task["input_json"])

        cur.execute("SELECT * FROM dna WHERE id = ?", (dna_id,))
        dna = dict(cur.fetchone())
        steps = json.loads(dna["steps_json"])
        total = len(steps)

        cur.execute(
            "UPDATE tasks SET status = 'running', total_steps = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (total, task_id),
        )
        conn.commit()

        pipeline_context = input_json.copy()
        overall_error = None

        for idx, step in enumerate(steps):
            organelle_name = step.get("organelle", "")
            step_name = step.get("step_name", f"step_{idx}")

            cur.execute(
                "UPDATE tasks SET current_step = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (idx, task_id),
            )
            conn.commit()

            step_input = pipeline_context.copy()
            step_input_json = json.dumps(step_input, ensure_ascii=False)

            result = self._run_step(
                conn, task_id, idx, organelle_name, step_name, step_input, dna_id
            )

            if result.error:
                self._fail_step(
                    conn, task_id, idx, step_input_json, result, step_name
                )
                overall_error = result.error
                break

            self._complete_step(
                conn, task_id, idx, step_input_json, result, step_name, organelle_name
            )

            if result.output:
                pipeline_context[f"{organelle_name}_output"] = result.output

        final_status = "failed" if overall_error else "completed"
        total_time = int((time.time() - json.loads(task["input_json"]).get("_start_time", time.time())) * 1000) if idx == total - 1 else 0

        cur.execute(
            """UPDATE tasks
               SET status = ?, error = ?, total_time_ms = ?, updated_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (final_status, overall_error or "", total_time, task_id),
        )
        conn.commit()
        conn.close()

        return {"task_id": task_id, "status": final_status, "error": overall_error}

    def _run_step(self, conn, task_id, idx, organelle_name, step_name, step_input, dna_id):
        cur = conn.cursor()
        cur.execute("SELECT * FROM organelles WHERE name = ?", (organelle_name,))
        organelle_row = cur.fetchone()

        if not organelle_row:
            return ExecutionResult(output="", error=f"Organelle '{organelle_name}' not found")

        organelle_data = dict(organelle_row)

        cur.execute(
            "SELECT * FROM mrna WHERE organelle_id = ? ORDER BY quality_score DESC LIMIT 1",
            (organelle_data["id"],),
        )
        mrna_row = cur.fetchone()
        if not mrna_row:
            return ExecutionResult(output="", error=f"No mRNA found for organelle '{organelle_name}'")

        mrna = dict(mrna_row)

        protein = self.ribosome.assemble(mrna["template"], step_input)

        context = OrganelleContext(
            organelle_id=organelle_data["id"],
            name=organelle_data["name"],
            capability=organelle_data["capability"],
            model=organelle_data["model"],
            api_base=organelle_data["api_base"],
            cost_per_token=organelle_data["cost_per_token"],
        )

        organelle_cls = ORGANELLE_MAP.get(organelle_name)
        if not organelle_cls:
            return ExecutionResult(output="", error=f"No class for organelle '{organelle_name}'")

        instance = organelle_cls(context)
        result = instance.run(protein, step_input)

        cur.execute(
            "UPDATE mrna SET usage_count = usage_count + 1 WHERE id = ?",
            (mrna["id"],),
        )
        conn.commit()

        return result

    def _fail_step(self, conn, task_id, idx, step_input_json, result, step_name):
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO step_results
               (task_id, step_index, input_json, output_json, protein_used,
                cost, time_ms, quality_score, status, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'failed', ?)""",
            (task_id, idx, step_input_json, result.output, step_name,
             result.cost, result.time_ms, result.quality_score, result.error),
        )
        conn.commit()

    def _complete_step(self, conn, task_id, idx, step_input_json, result, step_name, organelle_name):
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO step_results
               (task_id, step_index, input_json, output_json, protein_used,
                cost, time_ms, quality_score, status, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'completed', '')""",
            (task_id, idx, step_input_json, result.output, step_name,
             result.cost, result.time_ms, result.quality_score),
        )
        conn.commit()
