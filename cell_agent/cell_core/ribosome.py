from typing import Optional


class Ribosome:
    """Assembles proteins (prompts) from mRNA templates + task context."""

    def assemble(self, template: str, context: dict) -> str:
        protein = template
        for key, value in context.items():
            placeholder = "{" + key + "}"
            protein = protein.replace(placeholder, str(value))
        return protein
