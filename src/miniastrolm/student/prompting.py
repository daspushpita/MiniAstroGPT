# miniastrolm/student/prompting.py

from __future__ import annotations

def build_prompt(tokenizer, abstract: str) -> str:
    """
    Prompt used during inference.
    Must match the prefix of build_full_text exactly.
    """
    
    bos = tokenizer.bos_token or ""
    instruction_text = (
        "### Task: Explain the abstract in simple, non-technical language. "
        "Stay strictly on-topic."
    )
    input_text = f"### Abstract:\n{abstract}"
    
    return (
        f"{bos}"
        f"{instruction_text}\n\n"
        f"{input_text}\n\n"
        f"### Explanation:\n"
    )


def build_full_text(tokenizer, abstract: str, explanation: str) -> str:
    """
    Full training string (prompt + target).
    """
    eos = tokenizer.eos_token or ""
    return build_prompt(tokenizer, abstract) + explanation + eos