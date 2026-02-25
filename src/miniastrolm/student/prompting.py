from __future__ import annotations

import json


def build_prompt(tokenizer, paper_id: str, abstract: str) -> str:
    """
    Prompt used during inference.
    Must match the prefix of build_full_text exactly.
    """
    bos = tokenizer.bos_token or ""
    pid = "" if paper_id is None else str(paper_id)
    return (
        f"{bos}"
        "### Task: Explain the abstract in simple, non-technical language. "
        "Stay strictly on-topic."
        "### Task:\n"
        "Return ONE JSON object:\n"
        "{\"id\":\"<id>\",\"explanation\":\"...\"}\n\n"
        "Rules:\n"
        "- 180-220 words\n"
        "- Exactly 4 paragraphs separated by \\n\\n\n"
        "- Rewrite from scratch\n"
        "- No 'we present' or 'this paper'\n"
        "- Stay strictly anchored to the abstract\n\n"
        f"### ID:\n{pid}\n\n"
        f"### Abstract:\n{abstract}\n\n"
        "### Output:\n"
    )


def build_full_text(tokenizer, paper_id: str, abstract: str, explanation: str) -> str:
    """
    Full training string (prompt + JSON target).
    """
    eos = tokenizer.eos_token or ""
    target = json.dumps({"id": str(paper_id), "explanation": explanation}, ensure_ascii=False)
    return build_prompt(tokenizer, paper_id, abstract) + target + eos
