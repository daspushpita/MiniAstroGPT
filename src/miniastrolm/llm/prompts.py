from __future__ import annotations
from pathlib import Path

teacher_prompt_path = Path(__file__).parent.parent / "prompts" / "teacher_prompt_v1.txt"

class PromptLibrary:
    def __init__(self, teacher_prompt_text: str):
        self.teacher_prompt_text = teacher_prompt_text
        
    def format_abstract_block(self, 
                            i: int, 
                            paper_id: str, 
                            title: str, 
                            abstract: str) -> str:
        
        """Return the --- ABSTRACT i --- block exactly in your prompt.txt format."""
        return (
            f"--- ABSTRACT {i} ---\n"
            f"ID: {paper_id}\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}"
        )

    def build_teacher_prompt(self, items: list[dict]) -> str:
        """
        items: [{"id":..., "title":..., "abstract":...}, ...]
        returns: teacher_prompt_text + formatted blocks
        """
        blocks = []
        for idx, item in enumerate(items, start=1):
            block = self.format_abstract_block(
                i=idx,
                paper_id=item["id"],
                title=item.get("title", ""),
                abstract=item["abstract"],
            )
            blocks.append(block)
            all_blocks = self.teacher_prompt_text.rstrip() 
            + "\n\n" + "\n\n".join(blocks)
        return all_blocks

    def build_student_prompt(self, abstract: str) -> str:
        """
        Build the student prompt prefix.
        IMPORTANT: This exact format must be used during training.
        """
        return (
            "Rewrite the following astrophysics abstract into a clear, public-friendly explanation.\n"
            "Rewrite from scratch. Use simple language. Avoid academic-reporting phrases.\n"
            "No equations, symbols, or references to authors.\n\n"
            "ABSTRACT:\n"
            f"{abstract}\n\n"
            f"{self.EXPLANATION_MARKER}"
        )